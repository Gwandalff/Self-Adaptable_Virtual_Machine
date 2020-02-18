package miniJava.visitor.miniJava.impl;

import java.lang.Boolean;
import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class BooleanValueImpl extends ValueImpl implements BooleanValue {
	protected static final boolean VALUE_EDEFAULT = false;

	protected boolean value = VALUE_EDEFAULT;

	protected BooleanValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.BOOLEAN_VALUE;
	}

	public boolean isValue() {
		return value;
	}

	public void setValue(boolean newValue) {
		boolean oldValue = value;
		value = newValue;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.BOOLEAN_VALUE__VALUE, oldValue, value));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.BOOLEAN_VALUE__VALUE :
				return isValue();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.BOOLEAN_VALUE__VALUE :
				setValue((Boolean) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BOOLEAN_VALUE__VALUE :
				setValue(VALUE_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BOOLEAN_VALUE__VALUE :
				return value != VALUE_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__BooleanValue(this);
	}
}
