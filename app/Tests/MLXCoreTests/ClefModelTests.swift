import XCTest
@testable import MLXCore

final class ClefModelTests: XCTestCase {
    func testChoiceHelpMatchesTheRequestSchema() throws {
        let choice = LayaDecisionsPane.Question(name: "team", type: .choice, instructions: "Route it", criteria: "billing, sales")
        for clef in [false, true] {
            let example = LayaDecisionsPane.Question.choiceCriteriaExample(forClef: clef)
            let parsed = try XCTUnwrap(JSONSerialization.jsonObject(with: Data("{\(example)}".utf8)) as? NSDictionary)
            let criteria = try XCTUnwrap(choice.json(forClef: clef)?["criteria"])
            XCTAssertEqual(parsed, ["criteria": criteria] as NSDictionary)
        }
    }

    func testClefQuestionUsesObjectChoicesAndKeepsScoreLevels() throws {
        let choice = LayaDecisionsPane.Question(name: "route", type: .choice, instructions: "Route it", criteria: "sales, support, sales")
        let options = try XCTUnwrap(choice.json(forClef: true)?["criteria"] as? [String: NSNull])
        XCTAssertEqual(Set(options.keys), ["sales", "support"])
        XCTAssertEqual(choice.json(forClef: false)?["criteria"] as? [String], ["sales", "support", "sales"])
        let score = LayaDecisionsPane.Question(name: "urgency", type: .score, instructions: "", criteria: "low, high")
        XCTAssertEqual(score.json(forClef: true)?["criteria"] as? [String], ["low", "high"])
        let noul = LayaDecisionsPane.Question(name: "refund", type: .noul, instructions: "", criteria: "denied, allowed")
        XCTAssertEqual(noul.json(forClef: true)?["criteria"] as? [String: String], ["false": "denied", "true": "allowed"])
    }

    func testJointHeadOverridesQwenArchitecture() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "clef-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }
        for name in ["clef-4bit", "clef-8bit", "clef-flash-4bit", "clef-flash-8bit"] {
            let dir = (root as NSString).appendingPathComponent("mlx-community/\(name)")
            try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
            for (file, body) in ["config.json": #"{"model_type":"qwen3_5"}"#,
                                 "joint_head_config.json": "{}", "joint_head.safetensors": "x", "tokenizer.json": "{}"] {
                fm.createFile(atPath: (dir as NSString).appendingPathComponent(file), contents: Data(body.utf8))
            }
            fm.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"),
                          contents: Data(count: Int(DownloadManager.minimumWeightBytes) + 1))
            let models = DownloadManager.makeLocalModels(atDir: dir, displayName: name, idKey: name, source: .mlxServe)
            let model = try XCTUnwrap(models.first)
            XCTAssertEqual(model.modelType, "clef")
            XCTAssertNil(model.defect)
            XCTAssertTrue(model.isSupportedArchitecture)
            XCTAssertFalse(model.isChatPickable)
            XCTAssertTrue(isDecisionModelType(model.modelType))
        }
    }

    func testSearchAndDownloadRequireTheJointHead() throws {
        let row = HFModel(id: "mlx-community/clef-flash-4bit", downloads: 1, likes: 0, lastModified: nil,
                          tags: ["safetensors", "qwen3_5", "mlx", "clef", "text-classification"],
                          safetensors: nil, pipelineTag: "text-classification")
        XCTAssertEqual(row.mediaFamilyModelType, "clef")
        XCTAssertTrue(row.isSupportedArchitecture)
        let bundle = try XCTUnwrap(CustomMediaModels.bundle(arch: "clef", repoId: row.id))
        let files = ["config.json", "joint_head_config.json", "joint_head.safetensors", "tokenizer.json", "model.safetensors", "model.safetensors.index.json"]
            .map { HFSearchService.TreeFileEntry(path: $0, size: 1) }
        XCTAssertTrue(HFSearchService.mediaStructureSatisfied(markers: bundle.components[0].readyMarkers, files: files))
        XCTAssertFalse(HFSearchService.mediaStructureSatisfied(markers: bundle.components[0].readyMarkers,
                                                              files: files.filter { $0.path != "joint_head.safetensors" }))
        XCTAssertNil(MediaModality(modelType: "clef"))
    }
}
