import type { NextConfig } from "next";

// The FastAPI service (api/main.py). The browser only talks to this app; /api/* is
// proxied so there is no CORS setup and the API address stays server-side.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
