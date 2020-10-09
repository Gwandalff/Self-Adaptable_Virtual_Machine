package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.ArrayInstance;
import miniJava.interpreter.miniJava.ArrayRefValue;
import miniJava.interpreter.miniJava.ArrayTypeRef;
import miniJava.interpreter.miniJava.BooleanTypeRef;
import miniJava.interpreter.miniJava.BooleanValue;
import miniJava.interpreter.miniJava.ClassRef;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.IntegerTypeRef;
import miniJava.interpreter.miniJava.IntegerValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.NewArray;
import miniJava.interpreter.miniJava.NullValue;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.StringTypeRef;
import miniJava.interpreter.miniJava.TypeRef;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

public class NewArrayImpl extends ExpressionImpl implements NewArray {
	protected TypeRef type;

	protected Expression size;

	protected NewArrayImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NEW_ARRAY;
	}

	public TypeRef getType() {
		return type;
	}

	public NotificationChain basicSetType(TypeRef newType, NotificationChain msgs) {
		TypeRef oldType = type;
		type = newType;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_ARRAY__TYPE, oldType, newType);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setType(TypeRef newType) {
		if (newType != type) {
			NotificationChain msgs = null;
			if (type != null)
				msgs = ((InternalEObject) type).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.NEW_ARRAY__TYPE, null, msgs);
			if (newType != null)
				msgs = ((InternalEObject) newType).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.NEW_ARRAY__TYPE, null, msgs);
			msgs = basicSetType(newType, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_ARRAY__TYPE, newType, newType));
	}

	public Expression getSize() {
		return size;
	}

	public NotificationChain basicSetSize(Expression newSize, NotificationChain msgs) {
		Expression oldSize = size;
		size = newSize;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_ARRAY__SIZE, oldSize, newSize);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setSize(Expression newSize) {
		if (newSize != size) {
			NotificationChain msgs = null;
			if (size != null)
				msgs = ((InternalEObject) size).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.NEW_ARRAY__SIZE, null, msgs);
			if (newSize != null)
				msgs = ((InternalEObject) newSize).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.NEW_ARRAY__SIZE, null, msgs);
			msgs = basicSetSize(newSize, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_ARRAY__SIZE, newSize, newSize));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.NEW_ARRAY__TYPE :
				return basicSetType(null, msgs);
			case MiniJavaPackage.NEW_ARRAY__SIZE :
				return basicSetSize(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.NEW_ARRAY__TYPE :
				return getType();
			case MiniJavaPackage.NEW_ARRAY__SIZE :
				return getSize();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.NEW_ARRAY__TYPE :
				setType((TypeRef) newValue);
				return;
			case MiniJavaPackage.NEW_ARRAY__SIZE :
				setSize((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_ARRAY__TYPE :
				setType((TypeRef) null);
				return;
			case MiniJavaPackage.NEW_ARRAY__SIZE :
				setSize((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_ARRAY__TYPE :
				return type != null;
			case MiniJavaPackage.NEW_ARRAY__SIZE :
				return size != null;
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		ArrayInstance res = ((ArrayInstance) (MiniJavaFactory.eINSTANCE.createArrayInstance()));
		IntegerValue sizeArray = ((IntegerValue) (((IntegerValue) (((Expression) (this.getSize())).evaluateExpression((State) (state))))));
		res.setSize(sizeArray.getValue());
		state.getArraysHeap().add(res);
		Value defaultValue = ((Value) (null));
		if (this.getType() instanceof IntegerTypeRef) {
			IntegerValue idv = ((IntegerValue) (MiniJavaFactory.eINSTANCE.createIntegerValue()));
			idv.setValue(0);
			defaultValue = idv;
		}
		else {
			if (this.getType() instanceof BooleanTypeRef) {
				BooleanValue idv = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
				idv.setValue(false);
				defaultValue = idv;
			}
			else {
				if (this.getType() instanceof StringTypeRef) {
					NullValue idv = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
					defaultValue = idv;
				}
				else {
					if (((this.getType() instanceof ClassRef) || (this.getType() instanceof ArrayTypeRef))) {
						NullValue idv = ((NullValue) (MiniJavaFactory.eINSTANCE.createNullValue()));
						defaultValue = idv;
					}
				}
			}
		}
		int i = ((int) (0));
		int sz = ((int) (res.getSize()));
		while ((i) < (sz)) {
			Value dv = ((Value) (defaultValue));
			Value v = ((Value) (((Value) (dv)).copyj()));
			res.getValue().add(v);
			i = (i) + (1);
		}
		ArrayRefValue ret = ((ArrayRefValue) (MiniJavaFactory.eINSTANCE.createArrayRefValue()));
		ret.setInstance(res);
		result = (Value) (ret) ;
		return result;
	}
}
