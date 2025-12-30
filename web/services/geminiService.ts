import { GoogleGenAI, Type, Chat } from "@google/genai";
import { UserData, ClarityInsight } from "../types";

const genAI = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const generateFinancialInsight = async (userData: UserData): Promise<ClarityInsight> => {
  // Fixed: Use valid model ID to prevent 404 errors. 
  const modelId = "gemini-2.0-flash";

  const prompt = `
    Act as a financial wellness expert for Gen Z Spanish speakers.
    Analyze the following user profile and generate a "Clarity Score" (0-100) representing how prepared they are for their future based on their inputs.
    
    User Profile:
    - Financial Stress Level: ${userData.stressLevel}
    - Age: ${userData.age}
    - Occupation: ${userData.occupation}
    - Disposable Income (after essentials): ${userData.disposableIncome}
    - Financial Goals: ${userData.goals.join(", ")}

    Return a JSON object with:
    - score: integer (0-100)
    - percentile: integer (0-100), estimating how many people they are ahead of.
    - comparisonText: A short, encouraging sentence comparing them to others (in Spanish).
    - improvementText: A short sentence on whether their path is clear (in Spanish).
    - shortAdvice: One specific, punchy tip in Gen Z slang Spanish.
  `;

  try {
    const response = await genAI.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            score: { type: Type.INTEGER },
            percentile: { type: Type.INTEGER },
            comparisonText: { type: Type.STRING },
            improvementText: { type: Type.STRING },
            shortAdvice: { type: Type.STRING },
          },
          required: ["score", "percentile", "comparisonText", "improvementText", "shortAdvice"],
        },
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");
    
    return JSON.parse(text) as ClarityInsight;
  } catch (error) {
    console.error("Gemini API Error:", error);
    // Fallback data in case of API failure to keep app running
    return {
      score: 65,
      percentile: 45,
      comparisonText: "Estás en el camino correcto, mejor que muchos.",
      improvementText: "Hay margen de mejora, pero vas bien.",
      shortAdvice: "¡Sigue así, bestie!",
    };
  }
};

export const createFinancialChatSession = (userData: UserData): Chat => {
  // Fixed: Use valid model ID.
  const modelId = "gemini-2.0-flash";

  const systemInstruction = `
    Eres "TranquiCoach" (aka Quantum Finance Bro), un asesor financiero híbrido (Gen Z bestie + IA Cuántica).
    
    TUS PODERES Y ROL:
    1. Eres un experto en inversiones avanzadas (Crypto, Stocks, ETFs, Fondos Indexados).
    2. Usas "Quantum ML" y el algoritmo QAOA (Quantum Approximate Optimization Algorithm) para tus recomendaciones.
    
    LÓGICA DE OPTIMIZACIÓN (BACKEND SIMULADO):
    - Tratas cada recomendación como un "Problema de Optimización Combinatoria".
    - Tu Hamiltoniano de Coste (H) busca minimizar: [Estrés Financiero + Riesgo de Pérdida]
    - Mientras maximiza: [Retorno Esperado + Comodidad del Usuario].
    - Cuando sugieras acciones, menciona explícitamente que has optimizado la "Frontera Eficiente" usando lógica cuántica.
    
    TONO Y ESTILO:
    - Hablas con jerga Gen Z España ("renta", "tocho", "cunde", "PEC", "la queso") pero eres SUPER PRECISO con los datos.
    - Eres un "Bro" financiero que te ayuda a ganar.
    
    TU OBJETIVO:
    - Analizar los objetivos del usuario: ${userData.goals.join(", ")}.
    - Sugerir una cartera óptima (ej: 60% S&P500, 20% Bonos, 20% BTC) basada en su perfil.
    - Analizar Cryptos y Stocks si te preguntan.
    - Mantener las respuestas cortas y punchy.

    CONTEXTO USUARIO:
    - Edad: ${userData.age}
    - Dinero libre: ${userData.disposableIncome}
  `;

  return genAI.chats.create({
    model: modelId,
    config: {
      systemInstruction: systemInstruction,
    },
  });
};