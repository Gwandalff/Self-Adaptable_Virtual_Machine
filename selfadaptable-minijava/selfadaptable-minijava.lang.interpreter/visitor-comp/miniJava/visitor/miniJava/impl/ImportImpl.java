package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.Import;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class ImportImpl extends MinimalEObjectImpl.Container implements Import {
	protected Set<IDynamicModule> modules;
	
	protected static final String IMPORTEDNAMESPACE_EDEFAULT = null;

	protected String importedNamespace = IMPORTEDNAMESPACE_EDEFAULT;

	protected ImportImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.IMPORT;
	}

	public String getImportedNamespace() {
		return importedNamespace;
	}

	public void setImportedNamespace(String newImportedNamespace) {
		String oldImportedNamespace = importedNamespace;
		importedNamespace = newImportedNamespace;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.IMPORT__IMPORTED_NAMESPACE, oldImportedNamespace, importedNamespace));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.IMPORT__IMPORTED_NAMESPACE :
				return getImportedNamespace();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.IMPORT__IMPORTED_NAMESPACE :
				setImportedNamespace((String) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.IMPORT__IMPORTED_NAMESPACE :
				setImportedNamespace(IMPORTEDNAMESPACE_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.IMPORT__IMPORTED_NAMESPACE :
				return IMPORTEDNAMESPACE_EDEFAULT == null ? importedNamespace != null : !IMPORTEDNAMESPACE_EDEFAULT.equals(importedNamespace);
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__Import(this);
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
