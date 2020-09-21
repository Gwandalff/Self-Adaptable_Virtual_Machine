package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MethodCall;
import miniJava.visitor.miniJava.MethodCall2;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class MethodCall2Impl extends CallImpl implements MethodCall2 {
	protected MethodCall methodcall;

	protected MethodCall2Impl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.METHOD_CALL2;
	}

	public MethodCall getMethodcall() {
		if (methodcall != null && methodcall.eIsProxy()) {
			InternalEObject oldMethodcall = (InternalEObject) methodcall;
			methodcall = (MethodCall) eResolveProxy(oldMethodcall);
			if (methodcall != oldMethodcall) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.METHOD_CALL2__METHODCALL, oldMethodcall, methodcall));
			}
		}
		return methodcall;
	}

	public MethodCall basicGetMethodcall() {
		return methodcall;
	}

	public void setMethodcall(MethodCall newMethodcall) {
		MethodCall oldMethodcall = methodcall;
		methodcall = newMethodcall;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL2__METHODCALL, oldMethodcall, methodcall));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL2__METHODCALL :
				if (resolve)
					return getMethodcall();
				return basicGetMethodcall();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL2__METHODCALL :
				setMethodcall((MethodCall) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL2__METHODCALL :
				setMethodcall((MethodCall) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL2__METHODCALL :
				return methodcall != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__MethodCall2(this);
	}
}
