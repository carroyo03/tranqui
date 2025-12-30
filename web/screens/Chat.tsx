import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { UserData, ChatMessage } from '../types';
import { createFinancialChatSession } from '../services/api';

interface ChatScreenProps {
  userData: UserData;
  onBack: () => void;
}

export const ChatScreen: React.FC<ChatScreenProps> = ({ userData, onBack }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatSessionRef = useRef<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chatSessionRef.current) {
      chatSessionRef.current = createFinancialChatSession(userData);
      const initialGreeting: ChatMessage = {
        id: 'init-1',
        role: 'model',
        text: `¿Qué pasa? Soy tu **Quantum Finance Bro**. ¿Analizamos tus inversiones o te monto una cartera ganadora? 🚀`
      };
      setMessages([initialGreeting]);
    }
  }, [userData]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!inputText.trim() || !chatSessionRef.current || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: inputText
    };

    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setIsLoading(true);

    try {
      const result = await chatSessionRef.current.sendMessage({ message: userMsg.text });
      const responseText = result.text || "Ni tan mal, pero no te he entendido del todo.";

      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: responseText
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat error", error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: "Qué movida... se me ha ido el wifi mental. ¿Repites?"
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50/50 animate-fade-in relative font-sans">
      {/* Header - Glassmorphism with Quantum Badge */}
      <div className="flex items-center justify-between px-6 py-4 bg-white/80 backdrop-blur-md border-b border-gray-100 z-10 sticky top-0 shadow-sm transition-all md:px-12 lg:px-24">
        <button
          onClick={onBack}
          aria-label="Volver atrás"
          className="w-10 h-10 -ml-2 rounded-full flex items-center justify-center hover:bg-gray-50 active:bg-gray-100 transition-colors text-gray-600 focus:outline-none"
        >
          <span className="material-symbols-outlined font-bold">arrow_back</span>
        </button>
        <div className="flex flex-col items-center">
          <span className="font-bold text-gray-900 tracking-tight text-lg">TranquiCoach</span>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-emerald-500 font-bold flex items-center gap-1.5 uppercase tracking-wider bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
              </span>
              En línea
            </span>
            <span className="text-[10px] text-indigo-500 font-bold flex items-center gap-1 uppercase tracking-wider bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-100 shadow-sm">
              <span className="material-symbols-outlined text-[10px]">science</span>
              Quantum Bro
            </span>
          </div>
        </div>
        <div className="w-8"></div>
      </div>

      {/* Messages Area - Responsive Container */}
      <div className="flex-1 overflow-y-auto px-5 pt-6 space-y-6 pb-40 scroll-smooth w-full max-w-5xl mx-auto">
        {messages.map((msg, index) => (
          <div
            key={msg.id}
            className={`flex flex-col w-full animate-fade-in-up ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className={`max-w-[95%] md:max-w-[85%] px-6 py-4 text-[15px] relative transition-all duration-200 leading-relaxed shadow-sm ${msg.role === 'user'
              ? 'bg-primary text-white rounded-[1.5rem] rounded-br-sm shadow-primary/20'
              : 'bg-white text-gray-800 border border-gray-100 rounded-[1.5rem] rounded-bl-sm prose prose-sm max-w-none shadow-sm'
              }`}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ node, ...props }) => <div className="overflow-x-auto my-3 rounded-xl border border-gray-100 shadow-sm"><table className="min-w-full divide-y divide-gray-100" {...props} /></div>,
                  thead: ({ node, ...props }) => <thead className="bg-gray-50/50" {...props} />,
                  th: ({ node, ...props }) => <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider" {...props} />,
                  td: ({ node, ...props }) => <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600 border-t border-gray-50" {...props} />,
                  p: ({ node, ...props }) => <p className={`mb-2 last:mb-0 leading-7 ${msg.role === 'user' ? 'text-white' : 'text-gray-700'}`} {...props} />,
                  strong: ({ node, ...props }) => <strong className="font-semibold" {...props} />,
                  ul: ({ node, ...props }) => <ul className="list-disc pl-4 mb-3 space-y-1.5 marker:text-gray-300" {...props} />,
                  ol: ({ node, ...props }) => <ol className="list-decimal pl-4 mb-3 space-y-1.5 marker:text-gray-300" {...props} />,
                  li: ({ node, ...props }) => <li className="" {...props} />,
                  a: ({ node, ...props }) => <a className="underline hover:text-blue-500 transition-colors decoration-blue-200 underline-offset-2" target="_blank" rel="noopener noreferrer" {...props} />,
                  code: ({ node, ...props }) => <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono text-gray-800" {...props} />,
                }}
              >
                {msg.text}
              </ReactMarkdown>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isLoading && (
          <div className="flex w-full justify-start animate-fade-in pl-1">
            <div className="bg-white border border-gray-100 px-4 py-3 rounded-2xl rounded-bl-sm shadow-sm flex gap-1.5 items-center">
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-[bounce_1.4s_infinite_0ms]"></div>
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-[bounce_1.4s_infinite_200ms]"></div>
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-[bounce_1.4s_infinite_400ms]"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Floating Pill */}
      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-white via-white/95 to-transparent pt-4 pb-2 z-20">
        <div className="w-full max-w-4xl mx-auto px-4">
          <div className="flex items-center gap-2 bg-white p-2 pl-5 rounded-full shadow-[0_8px_30px_rgb(0,0,0,0.08)] border border-gray-100 focus-within:ring-2 focus-within:ring-primary/10 transition-all transform focus-within:-translate-y-1 relative z-30">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Analiza BTC, monta una cartera..."
              className="flex-1 bg-transparent border-none focus:ring-0 text-gray-800 placeholder-gray-400 py-2.5 font-medium focus:outline-none"
              disabled={isLoading}
              autoComplete="off"
            />
            <button
              onClick={handleSend}
              disabled={!inputText.trim() || isLoading}
              aria-label="Enviar mensaje"
              className={`h-11 w-11 rounded-full flex items-center justify-center transition-all duration-300 ${inputText.trim() && !isLoading
                ? 'bg-primary text-white shadow-lg shadow-primary/30 rotate-0 hover:bg-primary-hover hover:scale-105'
                : 'bg-gray-100 text-gray-300 rotate-90 cursor-default'
                }`}
            >
              <span className="material-symbols-outlined text-[22px]">arrow_upward</span>
            </button>
          </div>
          {/* Disclaimer Footer */}
          <div className="text-center py-2 relative z-20">
            <p className="text-[10px] text-gray-400 font-medium">
              Tranqui no ofrece asesoramiento financiero regulado. Si tienes más preguntas, no dudes en contactar a un asesor financiero.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};