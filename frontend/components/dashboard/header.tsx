import { ShieldCheck } from "lucide-react";

export function Header() {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b bg-background/80 backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-xl font-bold tracking-tight">CopyCatch AI</h1>
          <p className="text-xs text-muted-foreground">Hybrid Integrity Defense System</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        {/* Placeholder for future nav or user profile */}
        <span className="text-sm text-muted-foreground">v1.0.0</span>
      </div>
    </header>
  );
}
