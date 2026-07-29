import SwiftUI
import AppKit

/// Per-domain favicons, fetched once and kept for the session.
///
/// Best-effort by design: plenty of sites don't serve `/favicon.ico` (they
/// declare `<link rel="icon">` instead), so a miss is normal and falls back to a
/// monogram. Every domain is attempted at most ONCE — a failed domain that
/// retried on each render would hammer it for as long as the chat stays open.
@MainActor
final class FaviconStore: ObservableObject {
    static let shared = FaviconStore()

    @Published private(set) var icons: [String: NSImage] = [:]
    private var attempted: Set<String> = []

    /// The icon if we have it, kicking off one fetch if we've never tried.
    func icon(for domain: String) -> NSImage? {
        if let cached = icons[domain] { return cached }
        guard !attempted.contains(domain), !domain.isEmpty else { return nil }
        attempted.insert(domain)
        Task { await load(domain) }
        return nil
    }

    private func load(_ domain: String) async {
        guard let url = URL(string: "https://\(domain)/favicon.ico") else { return }
        var request = URLRequest(url: url)
        // The chat shouldn't wait on decoration; a slow host just gets a
        // monogram.
        request.timeoutInterval = 5
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let image = NSImage(data: data), image.size.width > 0
        else { return }
        icons[domain] = image
    }
}

/// A domain's icon: its favicon once loaded, otherwise a colored monogram.
///
/// The monogram is derived from the domain, so it is stable across renders and
/// distinct between hosts — a row never flickers between two different-looking
/// placeholders while the real icon is in flight.
struct FaviconView: View {
    let domain: String
    var size: CGFloat = 16

    @ObservedObject private var store = FaviconStore.shared

    private var monogramColor: Color {
        // Deterministic hue from the domain — not `hashValue`, which is seeded
        // per process and would give the same site a different color each launch.
        let sum = domain.unicodeScalars.reduce(0) { $0 &+ Int($1.value) }
        return Color(hue: Double(sum % 360) / 360, saturation: 0.55, brightness: 0.75)
    }

    var body: some View {
        Group {
            if let icon = store.icon(for: domain) {
                Image(nsImage: icon)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                ZStack {
                    monogramColor
                    Text(String(domain.first ?? "?").uppercased())
                        .font(.system(size: size * 0.6, weight: .semibold))
                        .foregroundStyle(.white)
                }
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
    }
}

/// The collapsed "N source(s)" chip under a reply, expanding to the list.
///
/// Only appears when a turn actually searched the web. Collapsed by default
/// because the sources are a provenance check, not the answer — but they are one
/// click away, and each row opens the page.
struct WebSourcesChip: View {
    let sources: [WebSource]
    @State private var expanded = false

    /// Distinct domains, in order — the stacked icons on the chip. Capped
    /// because the chip's width is what keeps it out of the message's way.
    private var iconDomains: [String] {
        var seen = Set<String>()
        return sources.map(\.domain).filter { seen.insert($0).inserted }.prefix(3).map { $0 }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    // Overlapped icon stack, leftmost on top.
                    HStack(spacing: -6) {
                        ForEach(iconDomains, id: \.self) { domain in
                            FaviconView(domain: domain)
                                .overlay(Circle().stroke(Color(nsColor: .windowBackgroundColor), lineWidth: 1.5))
                        }
                    }
                    Text("\(sources.count) source\(sources.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .overlay(Capsule().stroke(Color.secondary.opacity(0.25), lineWidth: 1))
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .help("Pages this answer drew on")

            if expanded {
                Text("\(sources.count) WEB SOURCE\(sources.count == 1 ? "" : "S")")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .padding(.top, 2)
                VStack(spacing: 4) {
                    ForEach(sources) { source in
                        sourceRow(source)
                    }
                }
            }
        }
    }

    private func sourceRow(_ source: WebSource) -> some View {
        Button {
            if let url = URL(string: source.url) { NSWorkspace.shared.open(url) }
        } label: {
            HStack(spacing: 10) {
                FaviconView(domain: source.domain, size: 20)
                Text(source.title)
                    .font(.callout)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 8)
                Text(source.domain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color.secondary.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .help(source.url)
    }
}
