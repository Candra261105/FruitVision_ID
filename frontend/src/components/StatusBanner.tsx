import type { ReactNode } from "react";

type StatusBannerProps = {
  kind: "info" | "warning" | "error";
  children: ReactNode;
};

export function StatusBanner({ kind, children }: StatusBannerProps) {
  return (
    <div className={`status-banner status-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}
