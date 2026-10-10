import { t } from "../i18n/i18n";
export type ErrorKind = "network" |
  "auth" |
  "forbidden" |
  "http" |
  "timeout" |
  "aborted" |
  "protocol" |
  "unsupported";

class StudioError extends Error {
  kind: ErrorKind;
  status: number | undefined;
  constructor(kind: ErrorKind, message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.name = "StudioError";
  }
}
function normalizeBaseUrl(input: string): string {
  let url: URL;
  try {
    url = new URL(input.trim());
  } catch {
    throw new StudioError(
      "protocol",
      t("Enter a complete http:// or https:// server URL."),
    );
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new StudioError(
      "protocol",
      t(
        "Server URLs must use HTTP(S) without credentials, query strings or fragments.",
      ),
    );
  url.pathname = url.pathname.replace(/\/+$/, "").replace(/\/v1$/, "");
  return url.toString().replace(/\/+$/, "");
}
export type RequestOptions = { signal?: AbortSignal; timeoutMs?: number; };

export type ClientOptions = RequestOptions & { apiKey?: string; fetch?: typeof fetch; onRequestFinished?: (path: string, method: string) => void; };

export type OpenResponse = { response: Response; signal: AbortSignal; close(): void; };

const record = (v: unknown): Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
function serverMessage(v: unknown): string {
  const r = record(v),
    error = record(r.error);
  return typeof error.message === "string"
    ? error.message
    : typeof r.error === "string"
      ? r.error
      : typeof r.message === "string"
        ? r.message
        : "Server request failed.";
}
function abortError(signal: AbortSignal): StudioError {
  return signal.reason instanceof StudioError
    ? signal.reason
    : new StudioError("aborted", t("Request stopped."));
}
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError(signal);
}
async function readBody(response: Response, signal: AbortSignal, maxBytes = 64 * 1024 * 1024): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.body)
    throw new StudioError("protocol", t("Response body is missing."));
  const chunks: Uint8Array[] = [];
  let size = 0;
  const reader = response.body.getReader();
  const abort = () => {
    void reader.cancel().catch(() => {});
  };
  signal.addEventListener("abort", abort, { once: true });
  try {
    throwIfAborted(signal);
    for (;;) {
      const { value, done } = await reader.read();
      throwIfAborted(signal);
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes)
        throw new StudioError("protocol", t("Response is too large."));
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    return bytes;
  } finally {
    signal.removeEventListener("abort", abort);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
class Client {
  private options: ClientOptions;
  readonly baseUrl: string;
  constructor(url: string, options: ClientOptions = {}) {
    this.options = options;
    this.baseUrl = normalizeBaseUrl(url);
  }
  redact(message: string): string {
    return this.options.apiKey
      ? message.split(this.options.apiKey).join("[redacted]")
      : message;
  }
  error(error: unknown, signal?: AbortSignal): StudioError {
    if (signal?.aborted) return abortError(signal);
    if (error instanceof StudioError)
      return new StudioError(
        error.kind,
        this.redact(error.message),
        error.status,
      );
    return new StudioError(
      "network",
      t(
        "Network/CORS failure. Check the server URL, reachability, CORS and HTTPS mixed-content policy.",
      ),
    );
  }
  async open(path: string, init: RequestInit = {}, options: RequestOptions = {}): Promise<OpenResponse> {
    if (path !== "/metrics.json" && !/^\/(?!\/)[a-zA-Z0-9/_-]+$/.test(path))
      throw new StudioError("protocol", t("Invalid API route."));
    const controller = new AbortController();
    const signals = [this.options.signal, options.signal].filter(
      (s): s is AbortSignal => !!s,
    );
    const listeners = signals.map((s) => {
      const fn = () => controller.abort(abortError(s));
      s.addEventListener("abort", fn, { once: true });
      if (s.aborted) fn();
      return () => s.removeEventListener("abort", fn);
    });
    const timeout = options.timeoutMs ?? this.options.timeoutMs ?? 60_000;
    if (!Number.isFinite(timeout) || timeout <= 0) {
      listeners.forEach((fn) => fn());
      throw new StudioError("protocol", t("Timeout must be positive."));
    }
    const timer = setTimeout(
      () =>
        controller.abort(new StudioError("timeout", t("Request timed out."))),
      timeout,
    );
    let closed = false;
    const close = () => {
      if (closed) return;
      closed = true;
      clearTimeout(timer);
      listeners.forEach((fn) => fn());
      controller.abort();
      this.options.onRequestFinished?.(path, init.method ?? "GET");
    };
    try {
      throwIfAborted(controller.signal);
      const headers = new Headers(init.headers);
      if (this.options.apiKey)
        headers.set("Authorization", `Bearer ${this.options.apiKey}`);
      const response = await (this.options.fetch ?? fetch)(
        this.baseUrl + path,
        {
          ...init,
          headers,
          signal: controller.signal,
          redirect: "error",
          credentials: "omit",
          cache: "no-store",
        },
      );
      throwIfAborted(controller.signal);
      if (!response.ok) {
        const bytes = await readBody(response, controller.signal, 1024 * 1024);
        const text = new TextDecoder().decode(bytes);
        let message: string;
        try {
          message = serverMessage(JSON.parse(text));
        } catch {
          message = text || response.statusText;
        }
        const prefix =
          response.status === 401
            ? "Authentication required: "
            : response.status === 403
              ? "Forbidden (mlx-serve host-local route or access policy): "
              : `HTTP ${response.status}: `;
        throw new StudioError(
          response.status === 401
            ? "auth"
            : response.status === 403
              ? "forbidden"
              : "http",
          prefix + this.redact(message).slice(0, 1000),
          response.status,
        );
      }
      return { response, signal: controller.signal, close };
    } catch (error) {
      const result = this.error(error, controller.signal);
      close();
      throw result;
    }
  }
  async bytes(path: string, init: RequestInit = {}, options: RequestOptions = {}, maxBytes?: number): Promise<Uint8Array<ArrayBuffer>> {
    const opened = await this.open(path, init, options);
    try {
      return await readBody(opened.response, opened.signal, maxBytes);
    } catch (error) {
      throw this.error(error, opened.signal);
    } finally {
      opened.close();
    }
  }
  async json(path: string, options: RequestOptions = {}, body?: unknown): Promise<unknown> {
    const bytes = await this.bytes(
      path,
      body === undefined ? {} : jsonPost(body),
      options,
    );
    try {
      return JSON.parse(
        new TextDecoder("utf-8", { fatal: true }).decode(bytes),
      );
    } catch {
      throw new StudioError("protocol", t("Invalid JSON response."));
    }
  }
}
const jsonPost = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export { StudioError, normalizeBaseUrl, record, serverMessage, abortError, throwIfAborted, readBody, Client, jsonPost };
