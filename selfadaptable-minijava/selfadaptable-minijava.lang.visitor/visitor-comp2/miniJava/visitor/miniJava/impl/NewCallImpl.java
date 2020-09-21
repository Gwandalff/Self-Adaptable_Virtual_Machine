package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.NewCall;
import miniJava.visitor.miniJava.NewObject;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class NewCallImpl extends CallImpl implements NewCall {
	protected NewObject newz;

	protected NewCallImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NEW_CALL;
	}

	public NewObject getNewz() {
		if (newz != null && newz.eIsProxy()) {
			InternalEObject oldNewz = (InternalEObject) newz;
			newz = (NewObject) eResolveProxy(oldNewz);
			if (newz != oldNewz) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.NEW_CALL__NEWZ, oldNewz, newz));
			}
		}
		return newz;
	}

	public NewObject basicGetNewz() {
		return newz;
	}

	public void setNewz(NewObject newNewz) {
		NewObject oldNewz = newz;
		newz = newNewz;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NEW_CALL__NEWZ, oldNewz, newz));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.NEW_CALL__NEWZ :
				if (resolve)
					return getNewz();
				return basicGetNewz();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.NEW_CALL__NEWZ :
				setNewz((NewObject) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_CALL__NEWZ :
				setNewz((NewObject) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NEW_CALL__NEWZ :
				return newz != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__NewCall(this);
	}
}
