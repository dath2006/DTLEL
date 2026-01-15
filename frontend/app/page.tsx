import Link from "next/link";
import { Activity, Database, ScanSearch, Shield } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  return (
    <main className="flex flex-1 flex-col gap-4 p-4 lg:gap-8 lg:p-6">
      <div className="flex items-center">
        <h1 className="text-lg font-semibold md:text-2xl">Overview</h1>
      </div>
      
      <div className="grid gap-4 md:grid-cols-2 md:gap-8 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">+2 from last hour</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Flagged Items</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">High plagiarism risk</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:gap-8 lg:grid-cols-2 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader className="flex flex-row items-center">
             <div className="grid gap-2">
                <CardTitle>Recent Analysis</CardTitle>
                <p className="text-sm text-muted-foreground">Recent documents processed by the engine.</p>
             </div>
             <Button asChild size="sm" className="ml-auto gap-1">
                <Link href="/analysis">
                    View All
                    <Activity className="h-4 w-4" />
                </Link>
             </Button>
          </CardHeader>
          <CardContent>
             <div className="h-[200px] flex items-center justify-center text-muted-foreground border-dashed border-2 rounded">
                 Activity Chart Placeholder
             </div>
          </CardContent>
        </Card>

        <Card>
            <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4">
                <Button asChild className="w-full" size="lg">
                    <Link href="/analysis">
                        <ScanSearch className="mr-2 h-4 w-4" />
                        Start New Analysis
                    </Link>
                </Button>
                <Button asChild variant="outline" className="w-full">
                    <Link href="/data-pool">
                        <Database className="mr-2 h-4 w-4" />
                        Manage Data Pool
                    </Link>
                </Button>
            </CardContent>
        </Card>
      </div>
    </main>
  );
}
