import type { Metadata } from "next";
import { Oxanium, Sora } from "next/font/google";
import "./globals.css";

const display = Oxanium({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

const body = Sora({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "DraftSight — Prédiction de draft LoL Pro",
  description:
    "Prédis le vainqueur d'une game pro League of Legends à partir de la draft 5v5, entraîné sur l'historique complet des drafts pro.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" className={`${display.variable} ${body.variable} h-full antialiased`}>
      <body className="relative min-h-full flex flex-col">{children}</body>
    </html>
  );
}
