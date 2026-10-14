cask "mlx-serve" do
  version "26.10.2"
  sha256 "2f49992cb5aca1cd91b59093d1c9f7ecf6166e842312600e1d0cd1c2574ed60a"

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
