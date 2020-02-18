package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

public class ObjectRefValueImpl extends ValueImpl implements ObjectRefValue {
	protected ObjectInstance instance;

	protected ObjectRefValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.OBJECT_REF_VALUE;
	}

	public ObjectInstance getInstance() {
		if (instance != null && instance.eIsProxy()) {
			InternalEObject oldInstance = (InternalEObject) instance;
			instance = (ObjectInstance) eResolveProxy(oldInstance);
			if (instance != oldInstance) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE, oldInstance, instance));
			}
		}
		return instance;
	}

	public ObjectInstance basicGetInstance() {
		return instance;
	}

	public void setInstance(ObjectInstance newInstance) {
		ObjectInstance oldInstance = instance;
		instance = newInstance;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE, oldInstance, instance));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE :
				if (resolve)
					return getInstance();
				return basicGetInstance();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE :
				setInstance((ObjectInstance) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE :
				setInstance((ObjectInstance) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OBJECT_REF_VALUE__INSTANCE :
				return instance != null;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__ObjectRefValue(this);
	}
}
