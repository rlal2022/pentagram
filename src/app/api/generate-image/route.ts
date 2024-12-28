import { NextResponse } from "next/server";
import { put } from "@vercel/blob";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    const url = new URL("https://pentagram-app--flux-demo-generate.modal.run/");

    url.searchParams.set("prompt", text);

    console.log("Requesting URL", url.toString());

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        "X-API-Key": process.env.API_KEY || "",
        Accept: "image/jpeg",
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API Response: ", errorText);
      throw new Error(
        `HTTP error! status: ${response.status}, message: ${errorText}`
      );
    }

    const imageBuffer = await response.arrayBuffer();

    const filename = `${crypto.randomUUID()}.jpg`;

    const blob = await put(filename, imageBuffer, {
      access: "public",
      contentType: "image/jpeg",
    });

    return NextResponse.json({
      success: true,
      imageUrl: blob.url,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}
