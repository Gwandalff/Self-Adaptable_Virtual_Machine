package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.util.Collection;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.OutputStream;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.util.EDataTypeEList;

public class OutputStreamImpl extends MinimalEObjectImpl.Container implements OutputStream {
	protected EList<String> stream;

	protected OutputStreamImpl() {
		super();
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
}
