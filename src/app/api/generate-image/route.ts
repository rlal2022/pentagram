import { NextResponse } from "next/server";
import { put } from "@vercel/blob";
import crypto from "crypto";

// Add this for debugging
const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  console.warn("API_KEY environment variable is not set!");
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    // Log the API key (first few characters only, for debugging)
    console.log("API Key present:", !!API_KEY);
    if (API_KEY) {
      console.log("API Key preview:", API_KEY.substring(0, 4) + "...");
    }

    const url = new URL("https://pentagram-app--flux-demo-generate.modal.run/");
    url.searchParams.set("prompt", text);

    console.log("Requesting URL", url.toString());

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        "X-API-Key": API_KEY || "",
        Accept: "image/jpeg",
      },
    });

    // Log the response status and headers for debugging
    console.log("Response status:", response.status);
    console.log("Response headers:", Object.fromEntries(response.headers));

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API Response Error:", errorText);
      return NextResponse.json(
        {
          success: false,
          error: `HTTP error! status: ${response.status}, message: ${errorText}`,
        },
        { status: response.status }
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
    console.error("Error in POST handler:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to process request",
      },
      { status: 500 }
    );
  }
}
