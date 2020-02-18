package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.Context;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Symbol;
import miniJava.interpreter.miniJava.SymbolRef;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

public class SymbolRefImpl extends ExpressionImpl implements SymbolRef {
	protected Symbol symbol;

	protected SymbolRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SYMBOL_REF;
	}

	public Symbol getSymbol() {
		if (symbol != null && symbol.eIsProxy()) {
			InternalEObject oldSymbol = (InternalEObject) symbol;
			symbol = (Symbol) eResolveProxy(oldSymbol);
			if (symbol != oldSymbol) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.SYMBOL_REF__SYMBOL, oldSymbol, symbol));
			}
		}
		return symbol;
	}

	public Symbol basicGetSymbol() {
		return symbol;
	}

	public void setSymbol(Symbol newSymbol) {
		Symbol oldSymbol = symbol;
		symbol = newSymbol;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.SYMBOL_REF__SYMBOL, oldSymbol, symbol));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_REF__SYMBOL :
				if (resolve)
					return getSymbol();
				return basicGetSymbol();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_REF__SYMBOL :
				setSymbol((Symbol) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_REF__SYMBOL :
				setSymbol((Symbol) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_REF__SYMBOL :
				return symbol != null;
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = (Value) (((Value) (((Context) (((State) (state)).findCurrentContext())).findBinding((Symbol) (this.getSymbol())).getValue())).copyj()) ;
		return result;
	}
}
