export interface UserData {
  stressLevel: 'high' | 'medium' | 'low' | null;
  age: number;
  occupation: string;
  disposableIncome: string;
  goals: string[];
}

export interface ClarityInsight {
  score: number;
  percentile: number;
  comparisonText: string;
  improvementText: string;
  shortAdvice: string;
}

export enum AppStep {
  WELCOME = 0,
  LOGIN = 9,
  SIGNUP = 10,
  FORGOT_PASSWORD = 11,
  EMAIL_SENT = 12,
  STRESS_QUESTION = 1,
  SOCIAL_PROOF = 2,
  PROFILE_INPUT = 3,
  GOALS_SELECTION = 4,
  ANALYZING = 5,
  CLARITY_SCORE = 6,
  CHAT = 7,
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
}