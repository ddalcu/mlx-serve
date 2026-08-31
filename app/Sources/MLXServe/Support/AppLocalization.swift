import Foundation

/// Localizes strings that are produced by model types rather than directly by
/// SwiftUI's `Text`/`Button` initializers.
enum L10n {
    private static let koreanBundle: Bundle? = Bundle.main
        .path(forResource: "ko", ofType: "lproj")
        .flatMap(Bundle.init(path:))

    static func text(_ key: String) -> String {
        let isKoreanEdition = Bundle.main.bundleIdentifier?.hasSuffix(".ko") == true
        let prefersKorean = Locale.preferredLanguages.first?.hasPrefix("ko") == true
        if (isKoreanEdition || prefersKorean),
           let koreanBundle {
            return koreanBundle.localizedString(forKey: key, value: key, table: nil)
        }
        return Bundle.main.localizedString(forKey: key, value: key, table: nil)
    }

    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: text(key), locale: Locale.current, arguments: arguments)
    }
}
