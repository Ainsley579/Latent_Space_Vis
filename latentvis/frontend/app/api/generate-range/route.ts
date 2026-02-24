import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const min = Number(body?.min);
    const max = Number(body?.max);

    console.log(`[generate-range] min=${min}, max=${max}`);

    return NextResponse.json({ ok: true, min, max });
  } catch (error) {
    console.error("[generate-range] invalid request:", error);
    return NextResponse.json({ ok: false, error: "invalid_request" }, { status: 400 });
  }
}

