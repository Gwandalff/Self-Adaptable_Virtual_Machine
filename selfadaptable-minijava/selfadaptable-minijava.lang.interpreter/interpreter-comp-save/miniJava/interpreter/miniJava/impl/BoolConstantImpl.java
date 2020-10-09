package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import miniJava.interpreter.miniJava.BoolConstant;
import miniJava.interpreter.miniJava.BooleanValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class BoolConstantImpl extends ExpressionImpl implements BoolConstant {
	protected static final String VALUE_EDEFAULT = null;

	protected String value = VALUE_EDEFAULT;

	protected BoolConstantImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.BOOL_CONSTANT;
	}

	public String getValue() {
		return value;
	}

	public void setValue(String newValue) {
		String oldValue = value;
		value = newValue;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.BOOL_CONSTANT__VALUE, oldValue, value));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.BOOL_CONSTANT__VALUE :
				return getValue();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.BOOL_CONSTANT__VALUE :
				setValue((String) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BOOL_CONSTANT__VALUE :
				setValue(VALUE_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.BOOL_CONSTANT__VALUE :
				return VALUE_EDEFAULT == null ? value != null : !VALUE_EDEFAULT.equals(value);
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		BooleanValue ret = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		ret.setValue(EqualService.equals((this.value), ("true")));
		result = (Value) (ret) ;
		return result;
	}
}
