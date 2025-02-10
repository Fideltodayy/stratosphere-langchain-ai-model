import 'dotenv/config'; // Add this at the very top
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { readFile } from 'fs/promises';

try {
    const text = await readFile('./stratosphere.txt', 'utf-8');
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 50,
        separators: ['\n\n', '\n', ' ', '']
    });
    const output = await splitter.createDocuments([text]);

    const sbApiKey = process.env.VITE_SUPABASE_KEY;
    const sbUrl = process.env.VITE_SUPABASE_URL;
    const openAIApiKey = process.env.VITE_OPENAI_API_KEY;

    if (!sbApiKey || !sbUrl || !openAIApiKey) {
        throw new Error("Missing required environment variables");
    }

    const client = createClient(sbUrl, sbApiKey);

    await SupabaseVectorStore.fromDocuments(
        output, 
        new OpenAIEmbeddings({ openAIApiKey }), 
        {
            client,
            tableName: 'documents',
        }
    );

} catch (error) {
    console.error("Error:", error);
}