import XCTest
@testable import MLXCore

/// Clicking a model in the browser opens its card: README + a button to the
/// Hugging Face page. The pure parts live in `ModelCard` so they are testable
/// without a row.
final class ModelCardTests: XCTestCase {

    func testUrlsPointAtTheRepoPageAndItsReadme() {
        XCTAssertEqual(ModelCard.pageURL(repoId: "mlx-community/gemma-4-12b-it-4bit")?.absoluteString,
                       "https://huggingface.co/mlx-community/gemma-4-12b-it-4bit")
        XCTAssertEqual(ModelCard.readmeURL(repoId: "mlx-community/gemma-4-12b-it-4bit")?.absoluteString,
                       "https://huggingface.co/mlx-community/gemma-4-12b-it-4bit/resolve/main/README.md")
    }

    /// A local model's name is a repo id in three spellings: `org/repo`
    /// (ours, LM Studio, Osaurus, the HF cache), MTPLX's flat `Org--Name`,
    /// and a bare folder with no org — which is nobody's HF repo.
    func testLocalNameResolvesToARepoIdOrNothing() {
        XCTAssertEqual(ModelCard.repoId(localName: "org/repo"), "org/repo")
        XCTAssertEqual(ModelCard.repoId(localName: "Org--Name"), "Org/Name")
        XCTAssertNil(ModelCard.repoId(localName: "just-a-folder"))
        XCTAssertNil(ModelCard.repoId(localName: ""))
    }

    /// A README opens with YAML front matter (license, tags, base_model) that
    /// renders as a wall of dashes and colons; the card shows the prose under it.
    func testFrontMatterIsStripped() {
        let readme = "---\nlicense: apache-2.0\ntags:\n- mlx\n---\n\n# Model\n\nHello."
        XCTAssertEqual(ModelCard.stripFrontMatter(readme), "# Model\n\nHello.")
        XCTAssertEqual(ModelCard.stripFrontMatter("# No front matter"), "# No front matter")
        // An unterminated block is not front matter — keep everything.
        XCTAssertEqual(ModelCard.stripFrontMatter("---\nlicense: x\n# body"), "---\nlicense: x\n# body")
    }
}
