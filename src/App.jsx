import { useState, useEffect, useRef } from 'react';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { retriever } from './utils/retriever';
import { combineDocuments } from './utils/combineDocuments';
import { formatConvHistory } from './utils/formatConvHistory';

export default function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const conversationContainerRef = useRef(null);
    const convHistory = useRef([]);

    // Langchain setup
    const llm = new ChatOpenAI({ 
        openAIApiKey: import.meta.env.VITE_OPENAI_API_KEY,
        temperature: 0.5 
    });

    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(`
        Given some conversation history (if any) and a question, convert the question to a standalone question. 
        conversation history: {conv_history}
        question: {question} 
        standalone question:`
    );

    const answerPrompt = PromptTemplate.fromTemplate(`
        You are a helpful and enthusiastic support bot for Stratosphere ID. Answer questions based on the context and conversation history.
        context: {context}
        conversation history: {conv_history}
        question: {question}
        answer:`
    );

    const standaloneQuestionChain = standaloneQuestionPrompt
        .pipe(llm)
        .pipe(new StringOutputParser());

    const retrieverChain = RunnableSequence.from([
        prevResult => prevResult.standalone_question,
        retriever,
        combineDocuments
    ]);

    const answerChain = answerPrompt
        .pipe(llm)
        .pipe(new StringOutputParser());

    const chain = RunnableSequence.from([
        {
            standalone_question: standaloneQuestionChain,
            original_input: new RunnablePassthrough()
        },
        {
            context: retrieverChain,
            question: ({ original_input }) => original_input.question,
            conv_history: ({ original_input }) => original_input.conv_history
        },
        answerChain
    ]);

    useEffect(() => {
        if (conversationContainerRef.current) {
            conversationContainerRef.current.scrollTop = 
                conversationContainerRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const question = input.trim();
        setInput('');
        setIsLoading(true);

        // Add user message
        setMessages(prev => [...prev, { text: question, isUser: true }]);

        try {
            const response = await chain.invoke({
                question: question,
                conv_history: formatConvHistory(convHistory.current)
            });

            convHistory.current.push(question);
            convHistory.current.push(response);

            // Add AI response
            setMessages(prev => [...prev, { text: response, isUser: false }]);
        } catch (error) {
            setMessages(prev => [...prev, { 
                text: "Sorry, I'm having trouble connecting. Please try again later.",
                isUser: false
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className="chatbot-container">
            <div className="chatbot-header">
                <img 
                    src="/logo.png" 
                    className="logo" 
                    alt="Stratosphere ID Logo"
                />
                <p className="sub-heading">Knowledge Bank</p>
            </div>
            
            <div 
                ref={conversationContainerRef}
                className="chatbot-conversation-container"
            >
                {messages.map((message, index) => (
                    <div 
                        key={index}
                        className={`speech ${message.isUser ? 'speech-human' : 'speech-ai'}`}
                    >
                        {message.text}
                    </div>
                ))}
                {isLoading && (
                    <div className="speech speech-ai">...</div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="chatbot-input-container">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    disabled={isLoading}
                />
                <button 
                    type="submit" 
                    className="submit-btn"
                    disabled={isLoading}
                >
                    <img
                        src="/send.png"
                        className="send-btn-icon"
                        alt="Send message"
                    />
                </button>
            </form>
        </main>
    );
}