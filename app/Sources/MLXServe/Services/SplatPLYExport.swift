import Foundation

/// A splat GLB (`KHR_gaussian_splatting` on one POINTS primitive, what Idol returns) as the standard 3D Gaussian
/// Splatting `.ply` every splat viewer opens: binary little-endian, 17 float properties per Gaussian, opacity as a
/// logit and scale as a log. Pure, so a test pins the bytes. The GLB is glTF y-up; the PLY keeps the turn the
/// reference's `save_ply` applies (its +y points down in a viewer), which is 180 degrees about x from the GLB.
enum SplatPLYExport {
    private static let names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    /// The PLY bytes, or nil when `glb` is not a splat GLB this writer understands.
    static func ply(fromGLB glb: Data) -> Data? {
        guard glb.count >= 20, u32(glb, 0) == 0x4654_6C67 else { return nil }
        let jsonLen = Int(u32(glb, 12))
        guard 20 + jsonLen + 8 <= glb.count,
              let gltf = (try? JSONSerialization.jsonObject(with: glb.subdata(in: 20..<(20 + jsonLen)))) as? [String: Any],
              let accessors = gltf["accessors"] as? [[String: Any]],
              let views = gltf["bufferViews"] as? [[String: Any]],
              let prim = ((gltf["meshes"] as? [[String: Any]])?.first?["primitives"] as? [[String: Any]])?.first,
              (prim["mode"] as? Int) == 0,
              let attrs = prim["attributes"] as? [String: Int] else { return nil }
        let binStart = 20 + jsonLen + 8
        func floats(_ key: String, _ comps: Int) -> [Float]? {
            guard let idx = attrs[key], idx >= 0, idx < accessors.count else { return nil }
            let a = accessors[idx]
            guard a["componentType"] as? Int == 5126, let count = a["count"] as? Int,
                  let v = a["bufferView"] as? Int, v >= 0, v < views.count else { return nil }
            let view = views[v]
            let at = binStart + ((view["byteOffset"] as? Int) ?? 0) + ((a["byteOffset"] as? Int) ?? 0)
            guard at + count * comps * 4 <= glb.count else { return nil }
            return glb.withUnsafeBytes { raw in
                (0..<(count * comps)).map { raw.loadUnaligned(fromByteOffset: at + $0 * 4, as: Float.self) }
            }
        }
        guard let pos = floats("POSITION", 3), let rot = floats("KHR_gaussian_splatting:ROTATION", 4),
              let scale = floats("KHR_gaussian_splatting:SCALE", 3), let opacity = floats("KHR_gaussian_splatting:OPACITY", 1),
              let dc = floats("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", 3) else { return nil }
        let n = pos.count / 3
        guard rot.count == n * 4, scale.count == n * 3, opacity.count == n, dc.count == n * 3 else { return nil }

        var out = Data("ply\nformat binary_little_endian 1.0\nelement vertex \(n)\n".utf8)
        for name in names { out.append(Data("property float \(name)\n".utf8)) }
        out.append(Data("end_header\n".utf8))
        var floatsOut = [Float]()
        floatsOut.reserveCapacity(n * names.count)
        let eps: Float = 1e-6
        for i in 0..<n {
            // (x, y, z) -> (x, -y, -z); the rotation (glTF x, y, z, w) premultiplied by 180 degrees about x.
            let qx = rot[i * 4], qy = rot[i * 4 + 1], qz = rot[i * 4 + 2], qw = rot[i * 4 + 3]
            let o = min(max(opacity[i], eps), 1 - eps)
            floatsOut += [pos[i * 3], -pos[i * 3 + 1], -pos[i * 3 + 2], 0, 0, 0,
                          dc[i * 3], dc[i * 3 + 1], dc[i * 3 + 2], logf(o / (1 - o)),
                          logf(max(scale[i * 3], eps)), logf(max(scale[i * 3 + 1], eps)), logf(max(scale[i * 3 + 2], eps)),
                          -qx, qw, -qz, qy]
        }
        floatsOut.withUnsafeBytes { out.append(contentsOf: $0) }
        return out
    }

    private static func u32(_ d: Data, _ at: Int) -> UInt32 {
        d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: at, as: UInt32.self) }
    }
}
