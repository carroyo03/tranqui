import React, { useState, useEffect, Suspense } from 'react';
import { onAuthStateChanged } from 'firebase/auth';
import { ProgressBar } from './components/ProgressBar';
import { AppStep, UserData, ClarityInsight } from './types';
import { generateFinancialInsight } from './services/api';
import { auth } from './src/firebase';

// Lazy Load Screens for Performance (Code Splitting)
const Welcome = React.lazy(() => import('./screens/Welcome').then(m => ({ default: m.Welcome })));
const StressQuestion = React.lazy(() => import('./screens/StressQuestion').then(m => ({ default: m.StressQuestion })));
const SocialProof = React.lazy(() => import('./screens/SocialProof').then(m => ({ default: m.SocialProof })));
const ProfileInput = React.lazy(() => import('./screens/ProfileInput').then(m => ({ default: m.ProfileInput })));
const GoalsSelection = React.lazy(() => import('./screens/GoalsSelection').then(m => ({ default: m.GoalsSelection })));
const ClarityScore = React.lazy(() => import('./screens/ClarityScore').then(m => ({ default: m.ClarityScore })));
const ChatScreen = React.lazy(() => import('./screens/Chat').then(m => ({ default: m.ChatScreen })));
const Login = React.lazy(() => import('./screens/Login').then(m => ({ default: m.Login })));
const Signup = React.lazy(() => import('./screens/Signup').then(m => ({ default: m.Signup })));
const ForgotPassword = React.lazy(() => import('./screens/ForgotPassword').then(m => ({ default: m.ForgotPassword })));
const EmailSent = React.lazy(() => import('./screens/EmailSent').then(m => ({ default: m.EmailSent })));

const TOTAL_STEPS = 6;

export default function App() {
  const [currentStep, setCurrentStep] = useState<AppStep>(AppStep.WELCOME);
  const [userData, setUserData] = useState<UserData>({
    stressLevel: null,
    age: 24,
    occupation: '',
    disposableIncome: '',
    goals: [],
  });
  const [insight, setInsight] = useState<ClarityInsight | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [user, setUser] = useState<any>(null);

  // Auth Listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      // If user is logged in and we are on auth screens, redirect?
      // For now we handle redirection explicitly in success handlers
      // to distinguish between Sign Up (Onboarding) and Sign In (Chat).
    });
    return () => unsubscribe();
  }, []);

  const goNext = () => {
    setCurrentStep((prev) => Math.min(prev + 1, AppStep.CLARITY_SCORE));
  };

  const goBack = () => {
    if (currentStep === AppStep.CHAT) {
      if (insight) {
        setCurrentStep(AppStep.CLARITY_SCORE);
      } else {
        // If logged in, maybe verify if we want to go back to welcome?
        // Let's stick to flow:
        setCurrentStep(AppStep.WELCOME);
      }
    } else if (currentStep === AppStep.LOGIN || currentStep === AppStep.SIGNUP) {
      setCurrentStep(AppStep.WELCOME);
    } else if (currentStep === AppStep.FORGOT_PASSWORD) {
      setCurrentStep(AppStep.LOGIN);
    } else if (currentStep === AppStep.EMAIL_SENT) {
      setCurrentStep(AppStep.LOGIN);
    } else {
      setCurrentStep((prev) => Math.max(prev - 1, AppStep.WELCOME));
    }
  };

  const handleStressUpdate = (level: UserData['stressLevel']) => {
    setUserData({ ...userData, stressLevel: level });
    goNext();
  };

  const handleProfileUpdate = (data: Partial<UserData>) => {
    setUserData({ ...userData, ...data });
    goNext();
  };

  const handleGoalsUpdate = async (goals: string[]) => {
    const updatedData = { ...userData, goals };
    setUserData(updatedData);

    setCurrentStep(AppStep.ANALYZING);
    setIsAnalyzing(true);

    const result = await generateFinancialInsight(updatedData);
    setInsight(result);
    setIsAnalyzing(false);
    setCurrentStep(AppStep.CLARITY_SCORE);
  };

  const handleStartChat = () => {
    setCurrentStep(AppStep.CHAT);
  };

  const handleLoginClick = () => setCurrentStep(AppStep.LOGIN);
  const handleSignupClick = () => setCurrentStep(AppStep.SIGNUP);
  const handleForgotPasswordClick = () => setCurrentStep(AppStep.FORGOT_PASSWORD);

  // Signup Success -> Onboarding
  const handleSignupSuccess = () => {
    setUserData({
      stressLevel: null,
      age: 24,
      occupation: '',
      disposableIncome: '',
      goals: [],
    });
    setCurrentStep(AppStep.STRESS_QUESTION);
  };

  // Login Success -> Chat
  const handleLoginSuccess = () => {
    // Optionally load existing user data here
    setUserData({
      stressLevel: 'medium',
      age: 27,
      occupation: 'Freelance',
      disposableIncome: '€1000+/mes', // Dummy data for now
      goals: ['Invertir (Crypto/Bolsa)', 'Fondo de emergencia'],
    });
    setInsight(null);
    setCurrentStep(AppStep.CHAT);
  };

  const handleForgotPasswordSubmit = () => setCurrentStep(AppStep.EMAIL_SENT);
  const handleBackToLogin = () => setCurrentStep(AppStep.LOGIN);

  // Render logic
  const renderScreen = () => {
    switch (currentStep) {
      case AppStep.WELCOME:
        // Changed onNext to point to handleSignupClick
        return <Welcome onNext={handleSignupClick} onLogin={handleLoginClick} />;
      case AppStep.LOGIN:
        return <Login
          onLoginSuccess={handleLoginSuccess}
          onGoToSignup={handleSignupClick}
          onForgotPassword={handleForgotPasswordClick}
          onBack={() => setCurrentStep(AppStep.WELCOME)}
        />;
      case AppStep.SIGNUP:
        return <Signup
          onSignupSuccess={handleSignupSuccess}
          onGoToLogin={handleLoginClick}
          onBack={() => setCurrentStep(AppStep.WELCOME)}
        />;
      case AppStep.FORGOT_PASSWORD:
        return <ForgotPassword onNext={handleForgotPasswordSubmit} onBack={handleBackToLogin} />;
      case AppStep.EMAIL_SENT:
        return <EmailSent onBackToLogin={handleBackToLogin} />;
      case AppStep.STRESS_QUESTION:
        return <StressQuestion onNext={handleStressUpdate} onBack={goBack} />;
      case AppStep.SOCIAL_PROOF:
        return <SocialProof stressLevel={userData.stressLevel} onNext={goNext} onBack={goBack} />;
      case AppStep.PROFILE_INPUT:
        return <ProfileInput onNext={handleProfileUpdate} onBack={goBack} initialData={userData} />;
      case AppStep.GOALS_SELECTION:
        return <GoalsSelection onNext={handleGoalsUpdate} onBack={goBack} initialGoals={userData.goals} />;
      case AppStep.ANALYZING:
        return (
          <div className="flex flex-col items-center justify-center h-full animate-fade-in">
            <div className="w-24 h-24 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-8"></div>
            <h2 className="text-xl font-bold text-gray-900">Analizando tu perfil...</h2>
            <p className="text-gray-500 mt-2">Nuestra IA está calculando tus oportunidades.</p>
          </div>
        );
      case AppStep.CLARITY_SCORE:
        return insight ? <ClarityScore insight={insight} onRestart={() => setCurrentStep(AppStep.WELCOME)} onChat={handleStartChat} /> : null;
      case AppStep.CHAT:
        return <ChatScreen userData={userData} onBack={goBack} />;
      default:
        return null;
    }
  };

  // Only show progress header for main questionnaire steps
  const showHeader = currentStep > AppStep.SIGNUP && currentStep < AppStep.CLARITY_SCORE && currentStep !== AppStep.ANALYZING && currentStep !== AppStep.FORGOT_PASSWORD && currentStep !== AppStep.EMAIL_SENT;

  return (
    <div className="flex justify-center min-h-screen bg-gray-50 font-sans text-gray-900">
      <div className="w-full h-[100dvh] bg-white flex flex-col relative overflow-hidden transition-all duration-300">

        {showHeader && (
          <header className="flex justify-between items-center px-6 py-5 border-b border-gray-100 z-10 bg-white/80 backdrop-blur-sm sticky top-0 md:px-12 lg:px-24">
            <button
              onClick={goBack}
              className="flex items-center text-gray-900 hover:text-primary transition-colors rounded-full p-1 -ml-2"
            >
              <span className="material-symbols-outlined text-2xl">arrow_back</span>
            </button>
            <div className="w-full max-w-md mx-auto">
              <ProgressBar currentStep={currentStep} totalSteps={TOTAL_STEPS} />
            </div>
            <div className="w-8"></div> {/* Spacer for balance */}
          </header>
        )}

        <main className="flex-1 flex flex-col px-0 py-0 overflow-hidden relative">
          <React.Suspense fallback={
            <div className="flex flex-col items-center justify-center h-full animate-fade-in">
              <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-4"></div>
              <p className="text-gray-400 text-sm">Cargando...</p>
            </div>
          }>
            {renderScreen()}
          </React.Suspense>
        </main>
      </div>
    </div>
  );
}