package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Inferior;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class InferiorImpl extends ExpressionImpl implements Inferior {
	protected Expression left;

	protected Expression right;

	protected InferiorImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.INFERIOR;
	}

	public Expression getLeft() {
		return left;
	}

	public NotificationChain basicSetLeft(Expression newLeft, NotificationChain msgs) {
		Expression oldLeft = left;
		left = newLeft;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.INFERIOR__LEFT, oldLeft, newLeft);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setLeft(Expression newLeft) {
		if (newLeft != left) {
			NotificationChain msgs = null;
			if (left != null)
				msgs = ((InternalEObject) left).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.INFERIOR__LEFT, null, msgs);
			if (newLeft != null)
				msgs = ((InternalEObject) newLeft).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.INFERIOR__LEFT, null, msgs);
			msgs = basicSetLeft(newLeft, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.INFERIOR__LEFT, newLeft, newLeft));
	}

	public Expression getRight() {
		return right;
	}

	public NotificationChain basicSetRight(Expression newRight, NotificationChain msgs) {
		Expression oldRight = right;
		right = newRight;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.INFERIOR__RIGHT, oldRight, newRight);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setRight(Expression newRight) {
		if (newRight != right) {
			NotificationChain msgs = null;
			if (right != null)
				msgs = ((InternalEObject) right).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.INFERIOR__RIGHT, null, msgs);
			if (newRight != null)
				msgs = ((InternalEObject) newRight).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.INFERIOR__RIGHT, null, msgs);
			msgs = basicSetRight(newRight, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.INFERIOR__RIGHT, newRight, newRight));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.INFERIOR__LEFT :
				return basicSetLeft(null, msgs);
			case MiniJavaPackage.INFERIOR__RIGHT :
				return basicSetRight(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.INFERIOR__LEFT :
				return getLeft();
			case MiniJavaPackage.INFERIOR__RIGHT :
				return getRight();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.INFERIOR__LEFT :
				setLeft((Expression) newValue);
				return;
			case MiniJavaPackage.INFERIOR__RIGHT :
				setRight((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.INFERIOR__LEFT :
				setLeft((Expression) null);
				return;
			case MiniJavaPackage.INFERIOR__RIGHT :
				setRight((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.INFERIOR__LEFT :
				return left != null;
			case MiniJavaPackage.INFERIOR__RIGHT :
				return right != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Inferior(this);
	}
}
