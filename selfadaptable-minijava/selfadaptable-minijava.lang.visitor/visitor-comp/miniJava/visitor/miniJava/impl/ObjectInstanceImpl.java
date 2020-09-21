package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.visitor.miniJava.Clazz;
import miniJava.visitor.miniJava.FieldBinding;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.ObjectInstance;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;
import visitor.VisitorInterface;

public class ObjectInstanceImpl extends MinimalEObjectImpl.Container implements ObjectInstance {
	protected EList<FieldBinding> fieldbindings;

	protected Clazz type;

	protected ObjectInstanceImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.OBJECT_INSTANCE;
	}

	public EList<FieldBinding> getFieldbindings() {
		if (fieldbindings == null) {
			fieldbindings = new EObjectContainmentEList<FieldBinding>(FieldBinding.class, this, MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS);
		}
		return fieldbindings;
	}

	public Clazz getType() {
		if (type != null && type.eIsProxy()) {
			InternalEObject oldType = (InternalEObject) type;
			type = (Clazz) eResolveProxy(oldType);
			if (type != oldType) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.OBJECT_INSTANCE__TYPE, oldType, type));
			}
		}
		return type;
	}

	public Clazz basicGetType() {
		return type;
	}

	public void setType(Clazz newType) {
		Clazz oldType = type;
		type = newType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.OBJECT_INSTANCE__TYPE, oldType, type));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS :
				return ((InternalEList<?>) getFieldbindings()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS :
				return getFieldbindings();
			case MiniJavaPackage.OBJECT_INSTANCE__TYPE :
				if (resolve)
					return getType();
				return basicGetType();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS :
				getFieldbindings().clear();
				getFieldbindings().addAll((Collection<? extends FieldBinding>) newValue);
				return;
			case MiniJavaPackage.OBJECT_INSTANCE__TYPE :
				setType((Clazz) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS :
				getFieldbindings().clear();
				return;
			case MiniJavaPackage.OBJECT_INSTANCE__TYPE :
				setType((Clazz) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_INSTANCE__FIELDBINDINGS :
				return fieldbindings != null && !fieldbindings.isEmpty();
			case MiniJavaPackage.OBJECT_INSTANCE__TYPE :
				return type != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__ObjectInstance(this);
	}
}
