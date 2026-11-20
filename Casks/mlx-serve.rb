cask "mlx-serve" do
  version "26.11.3"
  sha256 "8609126ac89cc1d03cf5715692b7aed1add7cf8dbc928f9c793b610c2e67a98c"

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
