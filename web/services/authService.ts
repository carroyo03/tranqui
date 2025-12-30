import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signInWithPopup,
    signOut,
    sendPasswordResetEmail as firebaseSendPasswordResetEmail,
    AuthError
} from "firebase/auth";
import { auth, googleProvider} from "../src/firebase";

// Error mapping for better UX
const getErrorMessage = (error: AuthError) => {
    switch (error.code) {
        case 'auth/email-already-in-use':
            return 'Este correo ya está registrado.';
        case 'auth/invalid-email':
            return 'El correo no es válido.';
        case 'auth/user-not-found':
            return 'No existe cuenta con este correo.';
        case 'auth/wrong-password':
            return 'Contraseña incorrecta.';
        case 'auth/weak-password':
            return 'La contraseña es muy débil (mínimo 6 caracteres).';
        default:
            return 'Ocurrió un error. Inténtalo de nuevo.';
    }
};

export const registerWithEmail = async (email: string, pass: string) => {
    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, pass);
        return { user: userCredential.user, error: null };
    } catch (error: any) {
        return { user: null, error: getErrorMessage(error) };
    }
};

export const loginWithEmail = async (email: string, pass: string) => {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, pass);
        return { user: userCredential.user, error: null };
    } catch (error: any) {
        return { user: null, error: getErrorMessage(error) };
    }
};

export const loginWithGoogle = async () => {
    try {
        const result = await signInWithPopup(auth, googleProvider);
        return { user: result.user, error: null };
    } catch (error: any) {
        return { user: null, error: getErrorMessage(error) };
    }
};



export const logout = async () => {
    await signOut(auth);
};

export const sendPasswordReset = async (email: string) => {
    try {
        await firebaseSendPasswordResetEmail(auth, email);
        return { success: true, error: null };
    } catch (error: any) {
        return { success: false, error: getErrorMessage(error) };
    }
};
