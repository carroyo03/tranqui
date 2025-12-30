import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signInWithPopup,
    signOut,
    sendPasswordResetEmail,
    UserCredential,
    AuthError
} from "firebase/auth";
import { auth, googleProvider } from "../src/firebase";

// Helper to format error messages
const getErrorMessage = (error: AuthError) => {
    switch (error.code) {
        case 'auth/email-already-in-use':
            return 'Este correo ya está registrado.';
        case 'auth/invalid-email':
            return 'Correo electrónico no válido.';
        case 'auth/weak-password':
            return 'La contraseña es muy débil.';
        case 'auth/user-not-found':
        case 'auth/wrong-password':
            return 'Credenciales incorrectas.';
        case 'auth/popup-closed-by-user':
            return 'Inicio de sesión cancelado.';
        case 'auth/too-many-requests':
            return 'Demasiados intentos. Inténtalo más tarde.';
        default:
            return 'Ocurrió un error. Inténtalo de nuevo.';
    }
};

export const registerWithEmail = async (email: string, password: string) => {
    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        return { user: userCredential.user, error: null };
    } catch (error) {
        return { user: null, error: getErrorMessage(error as AuthError) };
    }
};

export const loginWithEmail = async (email: string, password: string) => {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        return { user: userCredential.user, error: null };
    } catch (error) {
        return { user: null, error: getErrorMessage(error as AuthError) };
    }
};

export const loginWithGoogle = async () => {
    try {
        const userCredential = await signInWithPopup(auth, googleProvider);
        return { user: userCredential.user, error: null };
    } catch (error) {
        return { user: null, error: getErrorMessage(error as AuthError) };
    }
};



export const logout = async () => {
    try {
        await signOut(auth);
        return { success: true, error: null };
    } catch (error) {
        return { success: false, error: getErrorMessage(error as AuthError) };
    }
};

export const sendPasswordReset = async (email: string) => {
    try {
        await sendPasswordResetEmail(auth, email);
        return { success: true, error: null };
    } catch (error) {
        return { success: false, error: getErrorMessage(error as AuthError) };
    }
};
