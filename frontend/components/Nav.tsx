"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/live", label: "Live" },
  { href: "/faces", label: "Faces" },
  { href: "/history", label: "History" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="nav">
      <span className="brand">Face Recognition</span>
      <nav>
        {links.map((l) => (
          <Link key={l.href} href={l.href} className={path.startsWith(l.href) ? "active" : ""}>
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
