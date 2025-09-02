import React from "react";
import { Card } from "@/components/ui/card";
console.log("GetFiles component mounted");
export default function GetFiles() {
    // props are globally injected by Chainlit
    const files = props.files || [];
    console.log("React component received files:", files);    return (
        <div className="space-y-2">
            {files.length === 0 ? (
                <Card className="p-4 bg-secondary text-secondary-foreground">
                     No files found.
                </Card>
            ) : (
                files.map((file, idx) => (
                    <Card key={idx} className="p-4">
                        <span className="text-sm font-medium text-foreground">{file}</span>
                    </Card>
                ))
            )}
        </div>
    );
}
