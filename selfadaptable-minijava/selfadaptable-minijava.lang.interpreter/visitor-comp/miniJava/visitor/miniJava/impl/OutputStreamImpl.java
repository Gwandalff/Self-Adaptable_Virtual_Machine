package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.OutputStream;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EDataTypeEList;

import visitor.IDynamicModule;
import visitor.VisitorInterface;

public class OutputStreamImpl extends MinimalEObjectImpl.Container implements OutputStream {
	protected Set<IDynamicModule> modules;
	
	protected EList<String> stream;

	protected OutputStreamImpl() {
		super();
		modules = new HashSet<IDynamicModule>();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.OUTPUT_STREAM;
	}

	public EList<String> getStream() {
		if (stream == null) {
			stream = new EDataTypeEList<String>(String.class, this, MiniJavaPackage.OUTPUT_STREAM__STREAM);
		}
		return stream;
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.OUTPUT_STREAM__STREAM :
				return getStream();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.OUTPUT_STREAM__STREAM :
				getStream().clear();
				getStream().addAll((Collection<? extends String>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OUTPUT_STREAM__STREAM :
				getStream().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.OUTPUT_STREAM__STREAM :
				return stream != null && !stream.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__OutputStream(this);
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
