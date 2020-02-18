package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.ArrayLength;
import miniJava.interpreter.miniJava.ArrayRefValue;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.IntegerValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

public class ArrayLengthImpl extends ExpressionImpl implements ArrayLength {
	protected Expression array;

	protected ArrayLengthImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ARRAY_LENGTH;
	}

	public Expression getArray() {
		return array;
	}

	public NotificationChain basicSetArray(Expression newArray, NotificationChain msgs) {
		Expression oldArray = array;
		array = newArray;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ARRAY_LENGTH__ARRAY, oldArray, newArray);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setArray(Expression newArray) {
		if (newArray != array) {
			NotificationChain msgs = null;
			if (array != null)
				msgs = ((InternalEObject) array).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ARRAY_LENGTH__ARRAY, null, msgs);
			if (newArray != null)
				msgs = ((InternalEObject) newArray).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ARRAY_LENGTH__ARRAY, null, msgs);
			msgs = basicSetArray(newArray, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ARRAY_LENGTH__ARRAY, newArray, newArray));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_LENGTH__ARRAY :
				return basicSetArray(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_LENGTH__ARRAY :
				return getArray();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_LENGTH__ARRAY :
				setArray((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_LENGTH__ARRAY :
				setArray((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_LENGTH__ARRAY :
				return array != null;
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		ArrayRefValue arrayRef = ((ArrayRefValue) (((Expression) (this.getArray())).evaluateExpression((State) (state))));
		int size = ((int) (arrayRef.getInstance().getSize()));
		IntegerValue tmp = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
		tmp.setValue(size);
		result = (Value) (tmp) ;
		return result;
	}
}
