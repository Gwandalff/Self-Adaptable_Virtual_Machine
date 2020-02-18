package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.AccessLevel;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class MemberImpl extends TypedDeclarationImpl implements Member {
	protected static final AccessLevel ACCESS_EDEFAULT = AccessLevel.PRIVATE;

	protected AccessLevel access = ACCESS_EDEFAULT;

	protected MemberImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.MEMBER;
	}

	public AccessLevel getAccess() {
		return access;
	}

	public void setAccess(AccessLevel newAccess) {
		AccessLevel oldAccess = access;
		access = newAccess == null ? ACCESS_EDEFAULT : newAccess;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.MEMBER__ACCESS, oldAccess, access));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.MEMBER__ACCESS :
				return getAccess();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.MEMBER__ACCESS :
				setAccess((AccessLevel) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.MEMBER__ACCESS :
				setAccess(ACCESS_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.MEMBER__ACCESS :
				return access != ACCESS_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Member(this);
	}
}
