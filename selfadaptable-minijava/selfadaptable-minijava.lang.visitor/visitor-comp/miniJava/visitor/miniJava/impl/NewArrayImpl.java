package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.NewArray;
import miniJava.visitor.miniJava.TypeRef;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

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

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__NewArray(this);
	}
}
