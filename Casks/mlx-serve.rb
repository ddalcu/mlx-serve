cask "mlx-serve" do
  version "26.10.1"
  sha256 "f2d9759bcc132aee57bd78f95b35ddf2a266c48097c049e663653ae3d10ab4ef"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLX-Serve.dmg"
  name "MLX-Serve"
  desc "Native LLM server with OpenAI and Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  livecheck do
    url :url
    strategy :github_latest
  end

  auto_updates true
  depends_on arch: :arm64
  depends_on macos: :tahoe

  app "MLX-Serve.app"

  zap trash: [
    "~/.mlx-serve",
    "~/Library/Caches/com.dalcu.mlx-core",
    "~/Library/HTTPStorages/com.dalcu.mlx-core",
    "~/Library/HTTPStorages/com.dalcu.mlx-core.binarycookies",
    "~/Library/Preferences/com.dalcu.mlx-core.plist",
    "~/Library/WebKit/com.dalcu.mlx-core",
  ]
end
