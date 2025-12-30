import { UserData, ClarityInsight, ChatMessage } from "../types";
// import { Chat } from "@google/genai"; // Keeping types if needed, or we mock them

// We'll mimic the Gemini Interface so we don't have to rewrite all screens immediately,
// or we can refactor the screens. Let's refactor the screens to be cleaner.

export const generateFinancialInsight = async (userData: UserData): Promise<ClarityInsight> => {
    // Call Python Backend
    try {
        // Map UserData to Optimization Request if needed, 
        // For now, let's assume the backend 'optimize' endpoint takes tickers.
        // But the new frontend provides 'Goals'. We need a way to map Goals -> Tickers.
        // For this MVP step, we will use a fixed set of tickers or infer them.

        // Let's send the profile to a new endpoint OR map it locally.
        // Mapping locally for simplicity:
        let tickers = ["SPY", "AGG"]; // Default
        if (userData.goals.some(g => g.toLowerCase().includes("crypto"))) {
            tickers.push("BTC-USD", "ETH-USD");
        }
        if (userData.goals.some(g => g.toLowerCase().includes("tecnología"))) {
            tickers.push("AAPL", "MSFT", "NVDA");
        }

        // Risk inference
        let risk_aversion = 0.5;
        if (userData.stressLevel === "high") risk_aversion = 0.9;
        if (userData.stressLevel === "low") risk_aversion = 0.1;

        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                tickers: tickers,
                risk_aversion: risk_aversion,
                qaoa_reps: 1
            })
        });

        if (!response.ok) throw new Error("Optimization failed");

        const data = await response.json();

        // Map Quantum Result to ClarityInsight format
        return {
            score: Math.round(data.qaoa.metrics.sharpe_ratio * 20), // Proxy score
            percentile: 85, // Mock for now
            comparisonText: `Tu cartera cuántica supera al método clásico por ${(data.gap * 100).toFixed(2)}%.`,
            improvementText: "He optimizado tu diversificación usando QAOA.",
            shortAdvice: "Mantenlo HODL y compra en la caída.",
            // Store full data in a hidden field or global state if possible, 
            // but for now we just return the visual insight
            rawResults: data
        } as any;

    } catch (e) {
        console.error(e);
        return {
            score: 50,
            percentile: 50,
            comparisonText: "Error conectando al Quantum Core.",
            improvementText: "Inténtalo de nuevo.",
            shortAdvice: "Revisa tu conexión."
        };
    }
};

// Chat Adapter
// We need a class that mimics the GoogleGenAI Chat interface loosely, 
// or simply refactor the ChatScreen to use this function.
export class QuantumChatSession {
    constructor(private userData: UserData) { }

    async sendMessage(params: { message: string }) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: params.message,
                context: this.userData // Sending profile context directly as object
            })
        });

        const data = await response.json();
        return { text: data.response };
    }
}

export const createFinancialChatSession = (userData: UserData) => {
    return new QuantumChatSession(userData);
};
