package miniJava.visitor.miniJava.impl;

import java.lang.Integer;
import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class ArrayInstanceImpl extends MinimalEObjectImpl.Container implements ArrayInstance {
	protected static final int SIZE_EDEFAULT = 0;

	protected int size = SIZE_EDEFAULT;

	protected EList<Value> value;

	protected ArrayInstanceImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ARRAY_INSTANCE;
	}

	public EList<Value> getValue() {
		if (value == null) {
			value = new EObjectContainmentEList<Value>(Value.class, this, MiniJavaPackage.ARRAY_INSTANCE__VALUE);
		}
		return value;
	}

	public int getSize() {
		return size;
	}

	public void setSize(int newSize) {
		int oldSize = size;
		size = newSize;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ARRAY_INSTANCE__SIZE, oldSize, size));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_INSTANCE__VALUE :
				return ((InternalEList<?>) getValue()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_INSTANCE__VALUE :
				return getValue();
			case MiniJavaPackage.ARRAY_INSTANCE__SIZE :
				return getSize();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_INSTANCE__VALUE :
				getValue().clear();
				getValue().addAll((Collection<? extends Value>) newValue);
				return;
			case MiniJavaPackage.ARRAY_INSTANCE__SIZE :
				setSize((Integer) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_INSTANCE__VALUE :
				getValue().clear();
				return;
			case MiniJavaPackage.ARRAY_INSTANCE__SIZE :
				setSize(SIZE_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_INSTANCE__VALUE :
				return value != null && !value.isEmpty();
			case MiniJavaPackage.ARRAY_INSTANCE__SIZE :
				return size != SIZE_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__ArrayInstance(this);
	}

	@Override
	public void attach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void detach(IDynamicModule dm) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void notifyDynamicModules() {
		// TODO Auto-generated method stub
	}
}
