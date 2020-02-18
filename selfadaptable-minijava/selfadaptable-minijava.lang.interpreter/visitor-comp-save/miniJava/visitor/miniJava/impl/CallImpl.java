package miniJava.visitor.miniJava.impl;

import java.lang.Override;
import miniJava.visitor.miniJava.Call;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

public abstract class CallImpl extends MinimalEObjectImpl.Container implements Call {
	protected CallImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.CALL;
	}
}
