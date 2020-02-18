package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.NamedElement;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class NamedElementImpl extends MinimalEObjectImpl.Container implements NamedElement {
	protected Set<IDynamicModule> modules;
	
	protected static final String NAME_EDEFAULT = null;

	protected String name = NAME_EDEFAULT;

	protected NamedElementImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.NAMED_ELEMENT;
	}

	public String getName() {
		return name;
	}

	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.NAMED_ELEMENT__NAME, oldName, name));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.NAMED_ELEMENT__NAME :
				return getName();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.NAMED_ELEMENT__NAME :
				setName((String) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NAMED_ELEMENT__NAME :
				setName(NAME_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.NAMED_ELEMENT__NAME :
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__NamedElement(this);
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
	public boolean notifyDynamicModulesBefore() {
		boolean out = true;
		for (IDynamicModule dm : modules) {
			out = out && dm.updateBefore(this);
		}
		return out;
	}

	@Override
	public void notifyDynamicModulesAfter() {
		for (IDynamicModule dm : modules) {
			dm.updateAfter(this);
		}
	}
}
