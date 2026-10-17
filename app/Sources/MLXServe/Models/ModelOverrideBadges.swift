import Foundation

/// One chip in My Models' Settings column: a setting this model carries that
/// differs from the default, or the speculation method it will run.
struct OverrideBadge: Equatable, Identifiable {
    /// Declaration order is display order: what decides where and how the model
    /// runs comes first, so the overflow cuts the prompt-level knobs.
    enum Kind: Int { case engine, speculation, kvCache, sampling, kwargs }

    let kind: Kind
    var text: String
    /// Opens with the category, so a "+N" tooltip still says what each line is.
    var help: String
    /// The setting changes the output distribution.
    var lossy = false
    /// The speculation stone drawn beside `text`.
    var gem: GemStone?

    var id: Int { kind.rawValue }
}

enum ModelOverrideBadges {
    static let maxVisible = 2

    /// `socket` is the speculation the server will dispatch (`DrafterGems.badge`),
    /// which shows without any override: a pack's own MTP head is not a setting.
    static func badges(_ o: ModelOverride, modelType: String, socket: SocketBadge) -> [OverrideBadge] {
        var out: [OverrideBadge] = []

        if ModelSettingsApply.offersNeuralEngine(modelType: modelType), let ane = o.ane {
            out.append(OverrideBadge(
                kind: .engine, text: ane ? "ANE" : "GPU",
                help: ane ? "Engine: Neural Engine" : "Engine: GPU (Neural Engine switched off for this model)"))
        } else if ModelSettingsApply.hasInt8PrefillRoute(modelType: modelType), o.int8Prefill == true {
            out.append(OverrideBadge(kind: .engine, text: "Int8", help: "Engine: lossy int8 prefill", lossy: true))
        }

        let lossyAcceptance = o.mtpAcceptance == .typical || o.mtpAcceptance == .tokenv3
        var spec: [String] = []
        if let stone = socket.stone { spec.append(stone.label + (socket.skull ? ", lossy acceptance" : "")) }
        if let d = o.drafter { spec.append("drafter " + (d == "off" || d == "auto" ? d : "custom")) }
        if let m = o.mtp { spec.append("MTP " + (m ? "on" : "off")) }
        if let a = o.mtpAcceptance { spec.append("acceptance " + a.rawValue) }
        if let g = o.mtpGreedyTail { spec.append("greedy tail " + (g ? "on" : "off")) }
        if let stone = socket.stone {
            out.append(OverrideBadge(kind: .speculation, text: stone.badge, help: "Speculation: " + spec.joined(separator: " · "),
                                     lossy: socket.skull, gem: stone))
        } else if !spec.isEmpty {
            let off = o.drafter == "off" || o.mtp == false
            out.append(OverrideBadge(kind: .speculation, text: L10n.text(off ? "Spec off" : "Spec"),
                                     help: "Speculation: " + spec.joined(separator: " · "), lossy: lossyAcceptance))
        }

        if o.kvQuant != nil || o.ctxSize != nil {
            let kv = o.kvQuant.map { $0 == .off ? "KV off" : "KV \($0.rawValue)-bit" }
            let ctx = o.ctxSize.map(contextLabel)
            let text = [kv, kv == nil ? ctx.map { "\($0) ctx" } : ctx].compactMap { $0 }.joined(separator: " · ")
            var help = [String]()
            if let kv = o.kvQuant { help.append("KV " + kv.label) }
            if let n = o.ctxSize { help.append("context \(n) tokens") }
            out.append(OverrideBadge(kind: .kvCache, text: text, help: "KV cache: " + help.joined(separator: " · ")))
        }

        if !o.generationDefaults.rules.isEmpty {
            out.append(OverrideBadge(kind: .sampling, text: L10n.text("Sampling"),
                                     help: "Sampling: " + o.generationDefaults.rules.keys.sorted().joined(separator: ", ")))
        }
        if !o.templateKwargs.isEmpty {
            let pairs = o.sortedKwargKeys.map { "\($0)=\(o.templateKwargs[$0] ?? "")" }
            out.append(OverrideBadge(kind: .kwargs, text: "Kwargs", help: "Kwargs: " + pairs.joined(separator: ", ")))
        }
        return out
    }

    /// What fits the column, and what the "+N" chip stands for.
    static func overflow(_ badges: [OverrideBadge]) -> (shown: [OverrideBadge], hidden: [OverrideBadge]) {
        (Array(badges.prefix(maxVisible)), Array(badges.dropFirst(maxVisible)))
    }

    /// 32768 → "32k", 100000 → "100k"; a size that is no round k stays exact.
    private static func contextLabel(_ n: Int) -> String {
        if n >= 1024, n % 1024 == 0 { return "\(n / 1024)k" }
        if n >= 1000, n % 1000 == 0 { return "\(n / 1000)k" }
        return "\(n)"
    }
}
