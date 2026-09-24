import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "k8s-docs-rag",
  description: "Answers from the official Kubernetes docs, with citations and an eval dashboard",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <header className="border-b border-line bg-surface">
          <nav className="mx-auto flex max-w-6xl items-baseline gap-6 px-4 py-3">
            <Link href="/" className="font-semibold tracking-tight">
              k8s-docs-rag
            </Link>
            <Link href="/" className="text-sm text-muted hover:text-text">
              Ask
            </Link>
            <Link href="/eval" className="text-sm text-muted hover:text-text">
              Eval results
            </Link>
            <span className="ml-auto hidden text-xs text-muted sm:inline">
              Kubernetes 1.35 – 1.37 · answers cite kubernetes.io
            </span>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
