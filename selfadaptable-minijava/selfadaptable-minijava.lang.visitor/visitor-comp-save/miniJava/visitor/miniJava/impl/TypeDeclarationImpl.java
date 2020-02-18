package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.visitor.miniJava.AccessLevel;
import miniJava.visitor.miniJava.Interface;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.TypeDeclaration;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.EObjectResolvingEList;
import org.eclipse.emf.ecore.util.InternalEList;
import visitor.VisitorInterface;

public class TypeDeclarationImpl extends NamedElementImpl implements TypeDeclaration {
	protected static final AccessLevel ACCESSLEVEL_EDEFAULT = AccessLevel.PRIVATE;

	protected AccessLevel accessLevel = ACCESSLEVEL_EDEFAULT;

	protected EList<Interface> implementz;

	protected EList<Member> members;

	protected TypeDeclarationImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.TYPE_DECLARATION;
	}

	public AccessLevel getAccessLevel() {
		return accessLevel;
	}

	public void setAccessLevel(AccessLevel newAccessLevel) {
		AccessLevel oldAccessLevel = accessLevel;
		accessLevel = newAccessLevel == null ? ACCESSLEVEL_EDEFAULT : newAccessLevel;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.TYPE_DECLARATION__ACCESS_LEVEL, oldAccessLevel, accessLevel));
	}

	public EList<Interface> getImplementz() {
		if (implementz == null) {
			implementz = new EObjectResolvingEList<Interface>(Interface.class, this, MiniJavaPackage.TYPE_DECLARATION__IMPLEMENTZ);
		}
		return implementz;
	}

	public EList<Member> getMembers() {
		if (members == null) {
			members = new EObjectContainmentEList<Member>(Member.class, this, MiniJavaPackage.TYPE_DECLARATION__MEMBERS);
		}
		return members;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.TYPE_DECLARATION__MEMBERS :
				return ((InternalEList<?>) getMembers()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.TYPE_DECLARATION__ACCESS_LEVEL :
				return getAccessLevel();
			case MiniJavaPackage.TYPE_DECLARATION__IMPLEMENTZ :
				return getImplementz();
			case MiniJavaPackage.TYPE_DECLARATION__MEMBERS :
				return getMembers();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.TYPE_DECLARATION__ACCESS_LEVEL :
				setAccessLevel((AccessLevel) newValue);
				return;
			case MiniJavaPackage.TYPE_DECLARATION__IMPLEMENTZ :
				getImplementz().clear();
				getImplementz().addAll((Collection<? extends Interface>) newValue);
				return;
			case MiniJavaPackage.TYPE_DECLARATION__MEMBERS :
				getMembers().clear();
				getMembers().addAll((Collection<? extends Member>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.TYPE_DECLARATION__ACCESS_LEVEL :
				setAccessLevel(ACCESSLEVEL_EDEFAULT);
				return;
			case MiniJavaPackage.TYPE_DECLARATION__IMPLEMENTZ :
				getImplementz().clear();
				return;
			case MiniJavaPackage.TYPE_DECLARATION__MEMBERS :
				getMembers().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.TYPE_DECLARATION__ACCESS_LEVEL :
				return accessLevel != ACCESSLEVEL_EDEFAULT;
			case MiniJavaPackage.TYPE_DECLARATION__IMPLEMENTZ :
				return implementz != null && !implementz.isEmpty();
			case MiniJavaPackage.TYPE_DECLARATION__MEMBERS :
				return members != null && !members.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__TypeDeclaration(this);
	}
}
