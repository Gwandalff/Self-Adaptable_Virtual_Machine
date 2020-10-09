package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.BooleanValue;
import miniJava.interpreter.miniJava.Equality;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.IntegerValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.NullValue;
import miniJava.interpreter.miniJava.ObjectRefValue;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.StringValue;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class EqualityImpl extends ExpressionImpl implements Equality {
	protected Expression left;

	protected Expression right;

	protected EqualityImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.EQUALITY;
	}

	public Expression getLeft() {
		return left;
	}

	public NotificationChain basicSetLeft(Expression newLeft, NotificationChain msgs) {
		Expression oldLeft = left;
		left = newLeft;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.EQUALITY__LEFT, oldLeft, newLeft);
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
				msgs = ((InternalEObject) left).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.EQUALITY__LEFT, null, msgs);
			if (newLeft != null)
				msgs = ((InternalEObject) newLeft).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.EQUALITY__LEFT, null, msgs);
			msgs = basicSetLeft(newLeft, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.EQUALITY__LEFT, newLeft, newLeft));
	}

	public Expression getRight() {
		return right;
	}

	public NotificationChain basicSetRight(Expression newRight, NotificationChain msgs) {
		Expression oldRight = right;
		right = newRight;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.EQUALITY__RIGHT, oldRight, newRight);
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
				msgs = ((InternalEObject) right).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.EQUALITY__RIGHT, null, msgs);
			if (newRight != null)
				msgs = ((InternalEObject) newRight).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.EQUALITY__RIGHT, null, msgs);
			msgs = basicSetRight(newRight, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.EQUALITY__RIGHT, newRight, newRight));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.EQUALITY__LEFT :
				return basicSetLeft(null, msgs);
			case MiniJavaPackage.EQUALITY__RIGHT :
				return basicSetRight(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.EQUALITY__LEFT :
				return getLeft();
			case MiniJavaPackage.EQUALITY__RIGHT :
				return getRight();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.EQUALITY__LEFT :
				setLeft((Expression) newValue);
				return;
			case MiniJavaPackage.EQUALITY__RIGHT :
				setRight((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.EQUALITY__LEFT :
				setLeft((Expression) null);
				return;
			case MiniJavaPackage.EQUALITY__RIGHT :
				setRight((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.EQUALITY__LEFT :
				return left != null;
			case MiniJavaPackage.EQUALITY__RIGHT :
				return right != null;
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		Value left = ((Value) (((Expression) (this.getLeft())).evaluateExpression((State) (state))));
		Value right = ((Value) (((Expression) (this.getRight())).evaluateExpression((State) (state))));
		boolean tmp = ((boolean) (false));
		if (left instanceof IntegerValue) {
			if (right instanceof IntegerValue) {
				IntegerValue ileft = ((IntegerValue) (left));
				IntegerValue iright = ((IntegerValue) (right));
				tmp = EqualService.equals((ileft.getValue()), (iright.getValue()));
			}
		}
		else {
			if (left instanceof StringValue) {
				if (right instanceof StringValue) {
					StringValue ileft = ((StringValue) (left));
					StringValue iright = ((StringValue) (right));
					tmp = EqualService.equals((ileft.getValue()), (iright.getValue()));
				}
			}
			else {
				if (left instanceof BooleanValue) {
					if (right instanceof BooleanValue) {
						BooleanValue ileft = ((BooleanValue) (left));
						BooleanValue iright = ((BooleanValue) (right));
						tmp = EqualService.equals((ileft.isValue()), (iright.isValue()));
					}
				}
				else {
					if (left instanceof NullValue) {
						if (right instanceof NullValue) {
							tmp = true;
						}
					}
					else {
						if (left instanceof ObjectRefValue) {
							if (right instanceof ObjectRefValue) {
								ObjectRefValue ileft = ((ObjectRefValue) (left));
								ObjectRefValue iright = ((ObjectRefValue) (right));
								tmp = EqualService.equals((ileft.getInstance()), (iright.getInstance()));
							}
						}
					}
				}
			}
		}
		BooleanValue tmpo = ((BooleanValue) (MiniJavaFactory.eINSTANCE.createBooleanValue()));
		tmpo.setValue(tmp);
		result = (Value) (tmpo) ;
		return result;
	}
}
