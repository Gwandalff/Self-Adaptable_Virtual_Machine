package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Symbol;
import org.eclipse.emf.ecore.EClass;

public class SymbolImpl extends TypedDeclarationImpl implements Symbol {
	protected SymbolImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.SYMBOL;
	}
}
