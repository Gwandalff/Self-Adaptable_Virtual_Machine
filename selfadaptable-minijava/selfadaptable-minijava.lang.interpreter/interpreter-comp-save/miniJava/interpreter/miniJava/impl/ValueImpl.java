package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import java.lang.String;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.LogService;

public class ValueImpl extends MinimalEObjectImpl.Container implements Value {
	protected ValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VALUE;
	}

	public String customToString() {
		String result;
		result = (String) ("ERROR customToString not overloaded") ;
		return result;
	}

	public Value copyj() {
		Value result;
		LogService.log("ERROR copyj not implemented");
		result = (Value) (null) ;
		return result;
	}
}
