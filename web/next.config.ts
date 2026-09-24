import type { NextConfig } from "next";

// Two builds:
// - default: the app next to the FastAPI service (api/main.py); /api/* is proxied so there
//   is no CORS setup and the API address stays server-side.
// - STATIC_EXPORT=1: a static site (out/) that reads the JSON from scripts/export_demo.py,
//   for hosting without a backend. BASE_PATH is the site's path prefix, e.g. /k8s-docs-rag.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";
const STATIC = process.env.STATIC_EXPORT === "1";
const BASE_PATH = process.env.BASE_PATH ?? "";

const nextConfig: NextConfig = STATIC
  ? {
      output: "export",
      basePath: BASE_PATH || undefined,
      trailingSlash: true,
      env: { NEXT_PUBLIC_STATIC: "1", NEXT_PUBLIC_BASE_PATH: BASE_PATH },
    }
  : {
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
      },
    };

export default nextConfig;
