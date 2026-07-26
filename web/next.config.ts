import type { NextConfig } from "next";

const repo = "lol-draft-predictor";
const isGhPages = process.env.GITHUB_ACTIONS === "true";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  basePath: isGhPages ? `/${repo}` : "",
  assetPrefix: isGhPages ? `/${repo}/` : undefined,
  trailingSlash: true,
};

export default nextConfig;
