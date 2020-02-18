package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Symbol;
import miniJava.interpreter.miniJava.SymbolBinding;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.util.BasicEMap;
import org.eclipse.emf.common.util.EMap;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

public class SymbolToSymbolBindingMapImpl extends MinimalEObjectImpl.Container implements BasicEMap.Entry<Symbol, SymbolBinding> {
	protected Symbol key;

	protected SymbolBinding value;

	protected int hash = -1;

	protected SymbolToSymbolBindingMapImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SYMBOL_TO_SYMBOL_BINDING_MAP;
	}

	public Symbol getTypedKey() {
		if (key != null && key.eIsProxy()) {
			InternalEObject oldKey = (InternalEObject) key;
			key = (Symbol) eResolveProxy(oldKey);
			if (key != oldKey) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY, oldKey, key));
			}
		}
		return key;
	}

	public Symbol basicGetTypedKey() {
		return key;
	}

	public void setTypedKey(Symbol newKey) {
		Symbol oldKey = key;
		key = newKey;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY, oldKey, key));
	}

	public SymbolBinding getTypedValue() {
		if (value != null && value.eIsProxy()) {
			InternalEObject oldValue = (InternalEObject) value;
			value = (SymbolBinding) eResolveProxy(oldValue);
			if (value != oldValue) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE, oldValue, value));
			}
		}
		return value;
	}

	public SymbolBinding basicGetTypedValue() {
		return value;
	}

	public void setTypedValue(SymbolBinding newValue) {
		SymbolBinding oldValue = value;
		value = newValue;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE, oldValue, value));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY :
				if (resolve)
					return getTypedKey();
				return basicGetTypedKey();
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE :
				if (resolve)
					return getTypedValue();
				return basicGetTypedValue();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY :
				setTypedKey((Symbol) newValue);
				return;
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE :
				setTypedValue((SymbolBinding) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY :
				setTypedKey((Symbol) null);
				return;
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE :
				setTypedValue((SymbolBinding) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__KEY :
				return key != null;
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE :
				return value != null;
		}
		return super.eIsSet(featureID);
	}

	public int getHash() {
		if (hash == -1) {
			Object theKey = getKey();
			hash = (theKey == null ? 0 : theKey.hashCode());
		}
		return hash;
	}

	public void setHash(int hash) {
		this.hash = hash;
	}

	public Symbol getKey() {
		return getTypedKey();
	}

	public void setKey(Symbol key) {
		setTypedKey(key);
	}

	public SymbolBinding getValue() {
		return getTypedValue();
	}

	public SymbolBinding setValue(SymbolBinding value) {
		SymbolBinding oldValue = getValue();
		setTypedValue(value);
		return oldValue;
	}

	public EMap<Symbol, SymbolBinding> getEMap() {
		EObject container = eContainer();
		return container == null ? null : (EMap<Symbol, SymbolBinding>)container.eGet(eContainmentFeature());
	}
}
