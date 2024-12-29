import { NextResponse } from "next/server";
import { put } from "@vercel/blob";
import crypto from "crypto";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    const apiKey = request.headers.get("X-API-KEY");
    console.log("Received API KEY:", apiKey); // Debugging log

    if (apiKey !== process.env.NEXT_API_KEY) {
      console.log("Unauthorized: Keys do not match"); // Debugging log
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const modalUrl = process.env.NEXT_MODAL_URL;
    if (!modalUrl) {
      console.log("Modal URL is not defined"); // Debugging log
      return NextResponse.json(
        { error: "Internal Server Error: modalUrl is not defined" },
        { status: 500 }
      );
    }

    const url = new URL(modalUrl);

    url.searchParams.set("prompt", text);

    console.log("Requesting URL", url.toString());

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        "X-API-Key": process.env.NEXT_API_KEY || "",
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
    console.error("Error processing request:", error);
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}
